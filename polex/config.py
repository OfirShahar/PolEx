"""
Hyperparameter presets found with Optuna (Sec. 6 of the paper).

- top1:  maximizes the quality of the single highest-ranked candidate (default).
- top3:  maximizes the best alignment among the 3 highest-ranked candidates.
- top10: maximizes the best alignment among the 10 highest-ranked candidates
         (suited for plugging PolEx into a global reconstruction framework).
"""

PRESETS = {
    'top1': {
        'min_edge_length': 25.0,
        'min_length_ratio': 0.95,
        'epsilon_ratio': 0.02,
        'angle_threshold_deg': 17.5,
        'smoothing_kernel_size': 3,
        'gap': 10.0,
    },
    'top3': {
        'min_edge_length': 10.0,
        'min_length_ratio': 0.85,
        'epsilon_ratio': 0.02,
        'angle_threshold_deg': 15.0,
        'smoothing_kernel_size': 3,
        'gap': 5.0,
    },
    'top10': {
        'min_edge_length': 10.0,
        'min_length_ratio': 0.65,
        'epsilon_ratio': 0.01,
        'angle_threshold_deg': 10.0,
        'smoothing_kernel_size': 9,
        'gap': 5.0,
    },
}

DEFAULT_PRESET = 'top1'

# Zero-padding (in pixels) added around each fragment so that warped content is not cropped.
DEFAULT_PAD = 300
