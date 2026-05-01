"""
MLP position classifier that operates on normalised COCO 17-point keypoints
for two athletes (input size = 17 × 2 athletes × 3 values = 102 features).
"""

import torch
import torch.nn as nn

from .labels import NUM_CLASSES

INPUT_SIZE = 17 * 2 * 3  # 102


class PositionClassifier(nn.Module):
    """
    Feed-forward MLP: keypoints → position class.

    Architecture mirrors the high-accuracy MLP reported in the tk1475
    reference pipeline (>97% on ViCoS 18-class set).
    """

    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        num_classes: int = NUM_CLASSES,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64]

        layers: list[nn.Module] = []
        prev = input_size
        for h in hidden_sizes:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "PositionClassifier":
        """Load a trained model from a checkpoint file."""
        checkpoint = torch.load(path, map_location=device)
        # Support both raw state-dict and wrapped checkpoint dicts
        state = checkpoint.get("model_state_dict", checkpoint)
        model = cls()
        model.load_state_dict(state)
        model.eval()
        return model

    def save(self, path: str, extra: dict | None = None) -> None:
        """Save model weights (and optional metadata) to *path*."""
        payload: dict = {"model_state_dict": self.state_dict()}
        if extra:
            payload.update(extra)
        torch.save(payload, path)
