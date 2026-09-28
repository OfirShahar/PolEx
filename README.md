# PolEx: Pairwise Alignment and Compatibility for Arbitrary Irregular and Eroded Image Fragments

Official implementation of **PolEx** (**Pol**ygonal approximation and **Ex**trapolation), from the paper

> **Pairwise Alignment and Compatibility for Arbitrary Irregular and Eroded Image Fragments**<br>
> Ofir Itzhak Shahar, Gur Elkin, Ohad Ben-Shahar<br>
> *Pattern Recognition (ICPR 2026 - Oral)*, Lecture Notes in Computer Science, Springer, 2026<br>
> [[Paper]](https://link.springer.com/chapter/10.1007/978-3-032-31438-3_11) [[Dataset]](https://bgu365.sharepoint.com/:u:/s/ICVL/IQAYcApw9768T5upBhgnRxKwAX8XhUXBFfkF4JTEBbOPjzk?e=Oa8kah)

Given two image fragments of arbitrary shape (possibly eroded along their boundaries), PolEx computes a ranked list of candidate rigid alignments of one fragment relative to the other, together with a pictorial compatibility score for each. It makes no assumptions about the fragments' shapes, sizes or content.

## Method overview

1. **Extrapolation (preprocessing, once per fragment).** Erosion removes content near fragment boundaries. The content is extrapolated a few pixels beyond each fragment's boundary with an inpainting model. The paper uses a frozen, pre-trained latent diffusion model, without fine-tuning. The extrapolated content outside the fragment is its *extrapolated band*.
2. **Adaptive configuration sampling.** Each fragment's mask is smoothed and approximated by a polygon (Ramer–Douglas–Peucker). Nearly collinear vertices are merged, and *augmented* edges are added that span short, locally-collinear edge triplets. Every pair of edges with similar lengths (length ratio ≥ γ) defines one candidate alignment: the source edge is rotated to face the target edge and placed at a gap `g` from it, which accounts for erosion.
3. **Pictorial scoring.** For each candidate, the source fragment is warped. The two extrapolated bands are compared in LAB color space over their intersection, using patches of random sizes aggregated with a p-norm. Placements where the original fragments overlap are penalized. Candidates are ranked by this dissimilarity; lower is better.

## Installation

```bash
git clone https://github.com/OfirShahar/PolEx.git
cd PolEx
conda create -n polex python=3.10 -y
conda activate polex
pip install -r requirements.txt
```

A CUDA GPU is recommended, but PolEx also runs on CPU (a few seconds per pair).

## Data

The dataset used in the paper is available [here](https://bgu365.sharepoint.com/:u:/s/ICVL/IQAYcApw9768T5upBhgnRxKwAX8XhUXBFfkF4JTEBbOPjzk?e=Oa8kah). It contains 1000 puzzles generated from synthetic fresco images, split into Voronoi fragments with Perlin-noise erosion, and adjacent fragment pairs with ground-truth poses. Each split is laid out as follows:

```
test/
├── testset.csv              # one row per pair (see below)
└── pieces/                  # fragments: RGBA PNGs, transparent outside the fragment
    ├── puzzle_0_piece_0.png
    └── ...
```

Each CSV row contains `pair_idx`, `puzzle_idx`, `target_piece_idx`, `src_piece_idx`, `target_piece_path` and `src_piece_path` (relative to the split directory), and each fragment's ground-truth pose (`{target,src}_gt_rotation`, `{target,src}_gt_translation_x`, `{target,src}_gt_translation_y`).

To use your own fragments, provide them as RGBA PNGs in which alpha > 0 marks the fragment. The code that generated the dataset's puzzles (Voronoi fragmentation, Perlin-noise erosion and pair extraction) is in [`puzzle_generation/`](puzzle_generation/); see its README for details.

## Step 1: Extrapolation (preprocessing)

`extrapolate.py` extrapolates every fragment in a folder. For every `<name>.png` it writes `<name>_ext.png`, which has the same size and frame as the fragment and keeps the original content unchanged. It adds an extrapolated band that is delimited by dilating the fragment mask with an elliptical structuring element of size `--px` × `--px`. The band therefore extends roughly `--px`/2 pixels beyond the fragment. Two inpainting methods are available.

### Default: OpenCV inpainting

```bash
python extrapolate.py --indir <dataset>/test/pieces --outdir <dataset>/test/extrapolated_pieces
```

This uses classical inpainting (Telea's method in OpenCV). It runs on a CPU in the PolEx environment, with no extra dependencies, and takes a fraction of a second per fragment.

### As in the paper: latent diffusion inpainting

The paper's results were obtained with the pre-trained inpainting model of [CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion) (Rombach et al., CVPR 2022). It runs in that repository's environment:

```bash
git clone https://github.com/CompVis/latent-diffusion.git
cd latent-diffusion
conda env create -f environment.yaml
conda activate ldm
wget -O models/ldm/inpainting_big/last.ckpt https://heibox.uni-heidelberg.de/f/4d9ac7ea40c64582b7c9/?dl=1
cd -

python extrapolate.py \
    --indir <dataset>/test/pieces \
    --outdir <dataset>/test/extrapolated_pieces \
    --method ldm --ldm_dir <path/to/latent-diffusion>
```

This takes about 30 seconds per fragment on a single consumer GPU, and each fragment only needs it once.

| Option | Default | Description |
|---|---|---|
| `--method` | `opencv` | `opencv` or `ldm` |
| `--px` | 20 | Size of the dilation element delimiting the extrapolated band (n<sub>px</sub>) |
| `--skip_existing` | off | Skip fragments whose extrapolation already exists (useful for resuming) |
| `--steps` | 150 | `ldm` only: number of DDIM sampling steps |
| `--dim` | 512 | `ldm` only: model input resolution. Fragments are resized by a single scale factor per folder so the largest one fits |
| `--seed` | none | `ldm` only: seed for the diffusion sampler |
| `--save_intermediate` | off | `ldm` only: also save the full inpainted square and the inpainting mask of each fragment |

### Using other inpainting models

PolEx only needs, for each fragment, an RGBA image in the fragment's frame that holds the original content plus plausible content in a band around it. Any inpainting or outpainting model can replace the two methods above, including more recent diffusion models. To add one, write a function in `extrapolate.py` that inpaints the region given by `inpainting_mask(image)`, then pass the result through `crop_to_dilated_fragment` and `paste_original`. `extrapolate_fragment_opencv` is a minimal example of this pattern.

## Step 2: Pairwise alignment

### Command line

```bash
python align_pair.py \
    --target <dataset>/test/pieces/puzzle_579_piece_0.png \
    --src    <dataset>/test/pieces/puzzle_579_piece_7.png \
    --target_ext <dataset>/test/extrapolated_pieces/puzzle_579_piece_0_ext.png \
    --src_ext    <dataset>/test/extrapolated_pieces/puzzle_579_piece_7_ext.png \
    --top_k 5 --output candidates.json --visualize candidates.png
```

This prints the top-k candidate alignments. With `--output` they are also saved as JSON, and with `--visualize` the top-k alignments are rendered side by side.

### Python API

```python
from polex import find_candidates, load_rgba, PRESETS

candidates = find_candidates(
    load_rgba("target.png"), load_rgba("src.png"),
    load_rgba("target_ext.png"), load_rgba("src_ext.png"),
    **PRESETS["top1"],          # optional; top1 is the default
)
best = candidates[0]
print(best["rotation"], best["translation_x"], best["translation_y"], best["score"])
```

Each candidate describes how to move the **source** fragment onto the fixed **target** fragment. The source is first rotated by `rotation` degrees about its alpha-centroid; a positive angle is clockwise as displayed, since the image y-axis points down. It is then translated by (`translation_x`, `translation_y`) pixels, in the target's image frame. `score` is the dissimilarity (lower is better). The matched polygon edges are returned in `target_edge` and `src_edge`. `polex.imaging.warp` applies a candidate transformation to an image tensor.

### Hyperparameters

The hyperparameters were optimized with Optuna for three use cases. `top1` is the default:

| Preset | Optimized for | `min_edge_length` | `min_length_ratio` (γ) | `epsilon_ratio` (α) | `angle_threshold_deg` | `smoothing_kernel_size` | `gap` (g) |
|---|---|---|---|---|---|---|---|
| `top1` | the single best-ranked candidate | 25 | 0.95 | 0.02 | 17.5 | 3 | 10 |
| `top3` | the best of the top-3 candidates | 10 | 0.85 | 0.02 | 15.0 | 3 | 5 |
| `top10` | the best of the top-10 candidates (e.g. as input to a global solver) | 10 | 0.65 | 0.01 | 10.0 | 9 | 5 |

- `min_edge_length`: polygon edges shorter than 1.5 × this value (px) are discarded.
- `min_length_ratio` (γ): two edges can be matched only if min(L<sub>t</sub>, L<sub>s</sub>) / max(L<sub>t</sub>, L<sub>s</sub>) ≥ γ. Lower values give more candidates, at a higher runtime.
- `epsilon_ratio` (α): Ramer–Douglas–Peucker tolerance, as a fraction of the contour perimeter.
- `angle_threshold_deg`: threshold for merging nearly collinear vertices and for creating augmented edges.
- `smoothing_kernel_size`: size of the Gaussian kernel that smooths the fragment mask before polygonization.
- `gap` (g): distance in pixels between matched edges, to compensate for erosion.

In `align_pair.py` and `evaluate.py`, select a preset with `--preset`, and override any single value with the flag of the same name (e.g. `--gap 0`).

## Step 3: Evaluation

```bash
python evaluate.py --data_dir <dataset>/test --preset top1 --output results/top1.csv
```

By default, `evaluate.py` looks for the extrapolations from Step 1 in `<data_dir>/extrapolated_pieces`; use `--extrapolations_dir` to point elsewhere. For every pair, it evaluates the top `--max_candidates` (default 10) ranked candidates against the ground truth and writes one CSV row per candidate. It then prints a best-of-top-k summary (for `--k 1 3 10`) and saves it to `results/top1_summary.csv`. To re-print the summary of an existing results file, run `python evaluate.py --summarize results/top1.csv`.

The metrics (Sec. 5 of the paper) anchor the target fragment at its ground-truth pose and place the source relative to it using the prediction:

- **Relative position score** (↑): the area shared by the predicted and ground-truth source fragments, divided by the source fragment's area.
- **RMSE(θ)** (↓): the rotation error in degrees. The `rotation_error` column is the absolute angle difference in [0, 360), which is the value reported in the paper's tables. `rotation_error_wrapped` gives min(d, 360 − d).
- **RMSE(T)** (↓): the distance in pixels between the centroids of the predicted and ground-truth source fragments.

Best-of-top-k selects, for each pair, the candidate with the highest relative position score among the k best-ranked ones.

The pictorial score samples random patch sizes, so scores (and occasionally rankings) vary slightly between runs. Use `--seed` for repeatable runs.

## Repository structure

```
polex/
├── alignment.py    # find_candidates: candidate enumeration and ranking (main entry point)
├── geometry.py     # polygonal approximation, augmented edges, edge-to-edge alignment
├── scoring.py      # extrapolated-band LAB dissimilarity (random patches, p-norm)
├── imaging.py      # tensor conversion, centroids, rigid warps
├── evaluation.py   # dataset loading and pairwise metrics
└── config.py       # hyperparameter presets
extrapolate.py      # Step 1: extrapolation preprocessing (OpenCV or latent diffusion)
align_pair.py       # Step 2: align a single pair of fragments
evaluate.py         # Step 3: evaluate on a dataset split
puzzle_generation/  # dataset generation: fragmentation, erosion, pair extraction (see its README)
```

## Citation

If you use this code or the dataset, please cite:

```bibtex
@inproceedings{shahar2026polex,
  title     = {Pairwise Alignment and Compatibility for Arbitrary Irregular and Eroded Image Fragments},
  author    = {Shahar, Ofir Itzhak and Elkin, Gur and Ben-Shahar, Ohad},
  booktitle = {Pattern Recognition (ICPR 2026)},
  series    = {Lecture Notes in Computer Science},
  pages     = {154--169},
  publisher = {Springer Nature Switzerland},
  year      = {2026},
  doi       = {10.1007/978-3-032-31438-3_11}
}
```

## Acknowledgments

This work was funded in part by the European Union's Horizon 2020 research and innovation programme under grant agreement No 964854 (the RePAIR project). The extrapolation step builds on [latent-diffusion](https://github.com/CompVis/latent-diffusion) (Rombach et al., CVPR 2022).
