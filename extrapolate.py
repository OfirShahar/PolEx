"""
Pictorial extrapolation of fragments (preprocessing step, Sec. 4.1 of the paper).

Every fragment (a PNG with transparency) in --indir is extrapolated beyond its boundary. The
output `<name>_ext.png` has the same size and frame as the input fragment: the original content
is preserved, and extrapolated content is added in a band around the fragment (the fragment mask
dilated with an elliptical structuring element of size --px x --px, i.e. roughly --px / 2 pixels
beyond the boundary).

Methods:
  opencv (default)  Classical inpainting (OpenCV, Telea). Fast, CPU only, no extra dependencies.
  ldm               The pre-trained latent-diffusion inpainting model of Rombach et al.
                    (CompVis/latent-diffusion, `inpainting_big`), as used in the paper. Requires the
                    latent-diffusion repository, its environment and checkpoint (see the README).

Pipeline of the `ldm` method, per fragment:
  1. Resize (one scale factor for the whole folder, so all fragments fit in --dim minus a margin)
     and center on a --dim x --dim canvas.
  2. Build the inpainting mask (everything outside a slightly eroded fragment mask) and
     pre-fill it with OpenCV's Telea inpainting.
  3. Run the diffusion model (DDIM, --steps steps) to inpaint the masked region.
  4. Crop the result to the dilated fragment mask, resize back to the original resolution,
     crop again by the dilated original mask, and paste back the original pixels.
The `opencv` method performs steps 2 and 4 directly at the original resolution.

Usage:
    python extrapolate.py --indir <pieces_dir> --outdir <extrapolated_pieces_dir>
    python extrapolate.py --indir <pieces_dir> --outdir <extrapolated_pieces_dir> \
        --method ldm --ldm_dir <path/to/latent-diffusion>
"""

import argparse
import contextlib
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm


def find_scale_factor(image_paths, target_dim, px):
    """Largest scale factor (<= 1) that fits every fragment in target_dim - 2*px."""
    scale_factor = 1.0
    for path in image_paths:
        h, w = cv2.imread(path, cv2.IMREAD_UNCHANGED).shape[:2]
        scale_factor = min(scale_factor, (target_dim - 2 * px) / max(h, w))
    return scale_factor


def resize_and_center(image, scale_factor, target_dim):
    """Resize a BGRA image and center it on a target_dim x target_dim transparent canvas."""
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    resized = cv2.resize(image, (new_w, new_h)) if scale_factor != 1.0 else image
    canvas = np.zeros((target_dim, target_dim, 4), dtype=np.uint8)
    y_offset = max(0, (target_dim - new_h) // 2)
    x_offset = max(0, (target_dim - new_w) // 2)
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas, x_offset, y_offset


def inpainting_mask(image):
    """255 where content should be generated: outside the fragment, whose mask is slightly eroded."""
    fragment = (image[:, :, 3] > 0).astype(np.uint8) * 255
    fragment = cv2.erode(fragment, np.ones((5, 5), np.uint8), iterations=1)
    return 255 - fragment


def crop_to_dilated_fragment(image, mask, px):
    """Keep only the fragment region dilated by a px x px ellipse (BGR or BGRA input); returns BGRA."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px, px))
    region = cv2.dilate(cv2.bitwise_not(mask), kernel)
    result = cv2.bitwise_and(image, image, mask=region)
    if result.shape[2] == 3:
        result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result[region == 0, 3] = 0
    return result


def paste_original(extrapolated, image):
    fragment = image[:, :, 3] != 0
    extrapolated[fragment] = image[fragment]
    return extrapolated


# -----------------------------------------------------------------------------
# OpenCV (classical) inpainting
# -----------------------------------------------------------------------------

def extrapolate_fragment_opencv(image, px, radius=5):
    mask = inpainting_mask(image)
    inpainted = cv2.inpaint(cv2.cvtColor(image, cv2.COLOR_BGRA2BGR), mask, radius, cv2.INPAINT_TELEA)
    return paste_original(crop_to_dilated_fragment(inpainted, mask, px), image)


# -----------------------------------------------------------------------------
# Latent-diffusion inpainting (as used in the paper)
# -----------------------------------------------------------------------------

def make_batch(image_rgb, mask, device):
    """Batch in the format expected by the LDM inpainting model (values in [-1, 1])."""
    image = torch.from_numpy(image_rgb.astype(np.float32)[None].transpose(0, 3, 1, 2) / 255.0)
    mask = (mask.astype(np.float32) / 255.0 >= 0.5).astype(np.float32)[None, None]
    mask = torch.from_numpy(mask)
    batch = {"image": image, "mask": mask, "masked_image": (1 - mask) * image}
    return {k: v.to(device) * 2.0 - 1.0 for k, v in batch.items()}


def load_ldm(ldm_dir, config_path, ckpt_path, device):
    sys.path.insert(0, os.path.abspath(ldm_dir))
    from omegaconf import OmegaConf
    from ldm.models.diffusion.ddim import DDIMSampler
    from ldm.util import instantiate_from_config

    config = OmegaConf.load(config_path or os.path.join(ldm_dir, "models/ldm/inpainting_big/config.yaml"))
    model = instantiate_from_config(config.model)
    ckpt = ckpt_path or os.path.join(ldm_dir, "models/ldm/inpainting_big/last.ckpt")
    model.load_state_dict(torch.load(ckpt, map_location="cpu")["state_dict"], strict=False)
    model = model.to(device)
    return model, DDIMSampler(model)


@torch.no_grad()
def extrapolate_fragment_ldm(model, sampler, image, scale_factor, target_dim, px, steps, device):
    # 1-2) Resize, center, and build the (pre-filled) inpainting input
    resized, x_offset, y_offset = resize_and_center(image, scale_factor, target_dim)
    mask = inpainting_mask(resized)
    prefilled = cv2.inpaint(cv2.cvtColor(resized, cv2.COLOR_BGRA2BGR), mask, 5, cv2.INPAINT_TELEA)
    batch = make_batch(cv2.cvtColor(prefilled, cv2.COLOR_BGR2RGB), mask, device)

    # 3) Diffusion inpainting
    c = model.cond_stage_model.encode(batch["masked_image"])
    cc = torch.nn.functional.interpolate(batch["mask"], size=c.shape[-2:])
    c = torch.cat((c, cc), dim=1)
    shape = (c.shape[1] - 1,) + c.shape[2:]
    samples, _ = sampler.sample(S=steps, conditioning=c, batch_size=c.shape[0], shape=shape, verbose=False)
    predicted = torch.clamp((model.decode_first_stage(samples) + 1.0) / 2.0, min=0.0, max=1.0)

    image_01 = torch.clamp((batch["image"] + 1.0) / 2.0, min=0.0, max=1.0)
    mask_01 = torch.clamp((batch["mask"] + 1.0) / 2.0, min=0.0, max=1.0)
    rect = (1 - mask_01) * image_01 + mask_01 * predicted
    rect = cv2.cvtColor((rect.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    extrapolated = crop_to_dilated_fragment(rect, mask, px)

    # 4) Back to the original resolution and frame
    h, w = image.shape[:2]
    if scale_factor != 1.0:
        extrapolated = cv2.resize(extrapolated, (int(extrapolated.shape[1] / scale_factor),
                                                 int(extrapolated.shape[0] / scale_factor)))
        x_offset, y_offset = int(x_offset / scale_factor), int(y_offset / scale_factor)
    extrapolated = extrapolated[y_offset:y_offset + h, x_offset:x_offset + w]
    extrapolated = crop_to_dilated_fragment(extrapolated, inpainting_mask(image), px)
    return paste_original(extrapolated, image), rect, mask


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--indir", required=True, help="folder with fragment PNGs (with transparency)")
    parser.add_argument("--outdir", required=True, help="output folder for <name>_ext.png files")
    parser.add_argument("--method", choices=["opencv", "ldm"], default="opencv", help="inpainting method")
    parser.add_argument("--px", type=int, default=20,
                        help="size of the elliptical dilation element delimiting the extrapolated band "
                             "(n_px in the paper); the band extends ~px/2 pixels beyond the fragment")
    parser.add_argument("--skip_existing", action="store_true", help="skip fragments whose output already exists")
    ldm_args = parser.add_argument_group("ldm method")
    ldm_args.add_argument("--ldm_dir", default=".", help="path to the cloned CompVis/latent-diffusion repository")
    ldm_args.add_argument("--config", default=None, help="default: <ldm_dir>/models/ldm/inpainting_big/config.yaml")
    ldm_args.add_argument("--ckpt", default=None, help="default: <ldm_dir>/models/ldm/inpainting_big/last.ckpt")
    ldm_args.add_argument("--steps", type=int, default=150, help="number of DDIM sampling steps")
    ldm_args.add_argument("--dim", type=int, default=512, help="model input resolution")
    ldm_args.add_argument("--scale_factor", type=float, default=None,
                          help="override the resize factor (default: computed from the largest fragment in --indir)")
    ldm_args.add_argument("--seed", type=int, default=None, help="random seed for the diffusion sampler")
    ldm_args.add_argument("--save_intermediate", action="store_true",
                          help="also save the full inpainted square and mask of every fragment to <outdir>/extrapolation_doc")
    opt = parser.parse_args()

    image_paths = sorted(p for p in glob.glob(os.path.join(opt.indir, "*.png")) if not p.endswith("_ext.png"))
    print(f"Found {len(image_paths)} fragments.")
    os.makedirs(opt.outdir, exist_ok=True)
    details = {"method": opt.method, "px": opt.px, "num_fragments": len(image_paths)}

    if opt.method == "ldm":
        scale_factor = opt.scale_factor or find_scale_factor(image_paths, opt.dim, opt.px)
        print(f"Scale factor: {scale_factor:.4f}")
        details.update({"scale_factor": scale_factor, "steps": opt.steps, "dim": opt.dim})
        if opt.seed is not None:
            torch.manual_seed(opt.seed)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model, sampler = load_ldm(opt.ldm_dir, opt.config, opt.ckpt, device)
        doc_dir = os.path.join(opt.outdir, "extrapolation_doc")
        if opt.save_intermediate:
            os.makedirs(doc_dir, exist_ok=True)

    scope = model.ema_scope() if opt.method == "ldm" else contextlib.nullcontext()
    with scope:
        for path in tqdm(image_paths):
            name = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(opt.outdir, name + "_ext.png")
            if opt.skip_existing and os.path.exists(out_path):
                continue
            image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if image.ndim != 3 or image.shape[2] != 4:
                print(f"Skipping {path}: expected a 4-channel image with transparency")
                continue

            if opt.method == "opencv":
                extrapolated = extrapolate_fragment_opencv(image, opt.px)
            else:
                extrapolated, rect, mask = extrapolate_fragment_ldm(model, sampler, image, scale_factor,
                                                                    opt.dim, opt.px, opt.steps, device)
                if opt.save_intermediate:
                    cv2.imwrite(os.path.join(doc_dir, name + "_rect.png"), rect)
                    cv2.imwrite(os.path.join(doc_dir, name + "_mask.png"), mask)
            cv2.imwrite(out_path, extrapolated)

    with open(os.path.join(opt.outdir, "extrapolation_details.json"), "w") as f:
        json.dump(details, f, indent=2)


if __name__ == "__main__":
    main()
