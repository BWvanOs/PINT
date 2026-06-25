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


def screen_blend_layer(
    rgb: np.ndarray,
    proc: np.ndarray,
    color_name: str,
    gain: float,
) -> np.ndarray:
    color_vec = np.asarray(
        parse_composite_color(color_name),
        dtype=np.float32,
    )

    layer = np.clip(
        proc[..., None] * gain * color_vec[None, None, :],
        0.0,
        1.0,
    )

    return 1.0 - (1.0 - rgb) * (1.0 - layer)

def parse_composite_color(colorName: str) -> tuple[float, float, float]:
    colorName = str(colorName).strip()

    # Preset colors from the PINT palette
    if colorName in COMPOSITE_PALETTE:
        return COMPOSITE_PALETTE[colorName]

    # Custom RGB hex color
    if colorName.startswith("#") and len(colorName) == 7:
        try:
            r = int(colorName[1:3], 16) / 255.0
            g = int(colorName[3:5], 16) / 255.0
            b = int(colorName[5:7], 16) / 255.0
            return r, g, b
        except Exception:
            return 1.0, 1.0, 1.0

    return 1.0, 1.0, 1.0