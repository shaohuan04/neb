"""A top-down schematic of the train and its two rails.

Operators think in physical layout — which rail, which end of the train —
not in channel numbers. This renders that view and highlights the rail the
model flagged.
"""

NEUTRAL = "#39415A"
AFFECTED = "#FF6B72"
HEALTHY = "#35D6A0"
BODY = "#1D2333"
BODY_EDGE = "#2A3142"
TEXT = "#E9EDF5"
MUTED = "#98A2B8"


def train_diagram(affected_side=None, n_cars=8):
    """affected_side: 'Side I', 'Side II', or None for an all-clear view."""
    car_w, gap, start_x = 92, 10, 96
    width = start_x + n_cars * (car_w + gap) + 18
    s1_color = AFFECTED if affected_side == "Side I" else (NEUTRAL if affected_side else HEALTHY)
    s2_color = AFFECTED if affected_side == "Side II" else (NEUTRAL if affected_side else HEALTHY)
    s1_weight = 9 if affected_side == "Side I" else 5
    s2_weight = 9 if affected_side == "Side II" else 5

    parts = [f'<svg viewBox="0 0 {width} 190" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto">']

    # Rails
    parts.append(f'<rect x="{start_x}" y="34" width="{n_cars * (car_w + gap)}" height="{s1_weight}" rx="3" fill="{s1_color}"/>')
    parts.append(f'<rect x="{start_x}" y="148" width="{n_cars * (car_w + gap)}" height="{s2_weight}" rx="3" fill="{s2_color}"/>')

    # Rail labels
    parts.append(f'<text x="12" y="34" fill="{s1_color}" font-size="15" font-weight="700" font-family="Inter,sans-serif">Side I</text>')
    parts.append(f'<text x="12" y="52" fill="{MUTED}" font-size="11" font-family="Inter,sans-serif">pos 1,3,5,7</text>')
    parts.append(f'<text x="12" y="150" fill="{s2_color}" font-size="15" font-weight="700" font-family="Inter,sans-serif">Side II</text>')
    parts.append(f'<text x="12" y="168" fill="{MUTED}" font-size="11" font-family="Inter,sans-serif">pos 2,4,6,8</text>')

    # Cars, each with an axle box against each rail
    for i in range(n_cars):
        x = start_x + i * (car_w + gap)
        parts.append(f'<rect x="{x}" y="64" width="{car_w}" height="74" rx="9" fill="{BODY}" stroke="{BODY_EDGE}" stroke-width="1.5"/>')
        parts.append(f'<text x="{x + car_w / 2}" y="106" fill="{MUTED}" font-size="12" font-weight="600" '
                     f'text-anchor="middle" font-family="Inter,sans-serif">Car {i + 1}</text>')
        for ax in (0, 1):
            bx = x + 16 + ax * (car_w - 46)
            parts.append(f'<rect x="{bx}" y="50" width="30" height="12" rx="3" fill="{s1_color}" opacity="0.85"/>')
            parts.append(f'<rect x="{bx}" y="140" width="30" height="12" rx="3" fill="{s2_color}" opacity="0.85"/>')

    parts.append(f'<text x="{start_x}" y="185" fill="{MUTED}" font-size="11" '
                 f'font-family="Inter,sans-serif">Direction of travel →</text>')
    parts.append("</svg>")
    return "".join(parts)
