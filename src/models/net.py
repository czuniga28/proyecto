import torch
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18


def cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    """
    Convert boxes from centre and size to corners.

    Args:
        boxes: Tensor of shape (..., 4) with (cx, cy, w, h).

    Returns:
        Tensor of shape (..., 4) with (x0, y0, x1, y1).
    """
    # Predicting centre and a positive size guarantees x1 >= x0 and y1 >= y0,
    # which GIoU needs; four free corners could come out inverted
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


class SliceNet(nn.Module):
    """
    ResNet-18 with two heads: whether the slice contains the ACL, and its box.

    The input is not normalized with ImageNet statistics: its channels are
    neighbouring slices, already z-scored per volume, not RGB.

    Args:
        pretrained: Whether to start from the IMAGENET1K_V1 weights.
        dropout: Dropout probability before each head.
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.2) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.cls_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 1))
        self.box_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, 4))

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """
        Args:
            x: Batch of shape (B, 3, H, W).

        Returns:
            'logit': Tensor of shape (B,), before the sigmoid.
            'box': Tensor of shape (B, 4) with (x0, y0, x1, y1), normalized
                like the boxes from SliceDataset. It may extend past [0, 1]:
                clamping here would zero the gradient at the border, so it is clamped only at prediction time, before denormalize_boxes.
        """
        features = self.backbone(x)
        # squeeze() would also drop the batch dimension when B == 1
        logit = self.cls_head(features).squeeze(-1)
        box = cxcywh_to_xyxy(torch.sigmoid(self.box_head(features)))
        return {'logit': logit, 'box': box}
