"""
Position label definitions for the 18-class ViCoS BJJ dataset.

The suffix _a1 / _a2 denotes which athlete is dominant / on top.
Athlete assignment is determined by detection order (sorted by bounding-box
x-coordinate left-to-right at classification time).
"""

# Ordered list — index == class id used by the classifier model
POSITION_LABELS: list[str] = [
    "standing",         # 0
    "takedown_a1",      # 1  – A1 executing the takedown
    "takedown_a2",      # 2  – A2 executing the takedown
    "open_guard_a1",    # 3  – A1 on top (passing open guard)
    "open_guard_a2",    # 4  – A2 on top
    "half_guard_a1",    # 5  – A1 on top
    "half_guard_a2",    # 6  – A2 on top
    "closed_guard_a1",  # 7  – A1 on top (inside closed guard)
    "closed_guard_a2",  # 8  – A2 on top
    "50_50",            # 9  – symmetric leg entanglement
    "side_control_a1",  # 10 – A1 on top
    "side_control_a2",  # 11 – A2 on top
    "mount_a1",         # 12 – A1 mounted on A2
    "mount_a2",         # 13 – A2 mounted on A1
    "back_a1",          # 14 – A1 has back control
    "back_a2",          # 15 – A2 has back control
    "turtle_a1",        # 16 – A1 defending in turtle
    "turtle_a2",        # 17 – A2 defending in turtle
]

# Human-readable strings shown in the UI
DISPLAY_NAMES: dict[str, str] = {
    "standing":        "Standing",
    "takedown_a1":     "Takedown (A→B)",
    "takedown_a2":     "Takedown (B→A)",
    "open_guard_a1":   "Open Guard",
    "open_guard_a2":   "Open Guard",
    "half_guard_a1":   "Half Guard",
    "half_guard_a2":   "Half Guard",
    "closed_guard_a1": "Closed Guard",
    "closed_guard_a2": "Closed Guard",
    "50_50":           "50/50 Guard",
    "side_control_a1": "Side Control",
    "side_control_a2": "Side Control",
    "mount_a1":        "Mount",
    "mount_a2":        "Mount",
    "back_a1":         "Back Mount",
    "back_a2":         "Back Mount",
    "turtle_a1":       "Turtle",
    "turtle_a2":       "Turtle",
}

# Hex colours used in the web timeline and overlays
POSITION_COLORS: dict[str, str] = {
    "standing":        "#4CAF50",
    "takedown_a1":     "#FF9800",
    "takedown_a2":     "#FF9800",
    "open_guard_a1":   "#2196F3",
    "open_guard_a2":   "#42A5F5",
    "half_guard_a1":   "#1976D2",
    "half_guard_a2":   "#1E88E5",
    "closed_guard_a1": "#0D47A1",
    "closed_guard_a2": "#1565C0",
    "50_50":           "#9C27B0",
    "side_control_a1": "#FF5722",
    "side_control_a2": "#FF7043",
    "mount_a1":        "#F44336",
    "mount_a2":        "#EF5350",
    "back_a1":         "#E91E63",
    "back_a2":         "#EC407A",
    "turtle_a1":       "#795548",
    "turtle_a2":       "#8D6E63",
}

NUM_CLASSES: int = len(POSITION_LABELS)
LABEL_TO_IDX: dict[str, int] = {lbl: i for i, lbl in enumerate(POSITION_LABELS)}
IDX_TO_LABEL: dict[int, str] = {i: lbl for i, lbl in enumerate(POSITION_LABELS)}
