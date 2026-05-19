import numpy as np

COMPOSITE_PALETTE = {
    "Cyan":    (0.00, 1.00, 1.00),
    "Magenta": (1.00, 0.00, 1.00),
    "Yellow":  (1.00, 1.00, 0.00),
    "Green":   (0.00, 1.00, 0.00),
    "Red":     (1.00, 0.00, 0.00),
    "Blue":    (0.00, 0.40, 1.00),
    "Orange":  (1.00, 0.55, 0.00),
    "White":   (1.00, 1.00, 1.00),
}

COMPOSITE_COLOR_CHOICES = list(COMPOSITE_PALETTE.keys())
MAX_COMPOSITE_CHANNELS = 8
COMPOSITE_EMPTY_CHOICE = ">> Leave blank <<"


def screen_blend_layer(rgb: np.ndarray, proc: np.ndarray, color_name: str, gain: float) -> np.ndarray:
    color_vec = np.asarray(COMPOSITE_PALETTE.get(color_name, (1.0, 1.0, 1.0)), dtype=np.float32)
    layer = np.clip(proc[..., None] * gain * color_vec[None, None, :], 0.0, 1.0)
    return 1.0 - (1.0 - rgb) * (1.0 - layer)