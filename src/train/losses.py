import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import generalized_box_iou_loss


def detection_loss(
    out: dict[str, Tensor],
    batch: dict[str, Tensor],
    pos_weight: float,
    box_weight: float = 1.0,
) -> tuple[Tensor, dict[str, float]]:
    """
    Classification loss on every slice plus box loss on the positive ones.

    Args:
        out: Output of SliceNet, with 'logit' (B,) and 'box' (B, 4).
        batch: Batch from SliceDataset, already on the model's device.
        pos_weight: Weight of the positive class in the BCE.
        box_weight: Weight of the box loss in the total.

    Returns:
        The total loss, and its 'cls' and 'box' terms as floats for logging.
    """
    # Created on the logit's device: a CPU tensor here fails on MPS
    cls = F.binary_cross_entropy_with_logits(
        out['logit'],
        batch['label'],
        pos_weight=torch.tensor(pos_weight, device=out['logit'].device),
    )

    mask = batch['has_box']
    if mask.any():
        pred, gt = out['box'][mask], batch['box'][mask]
        box = F.l1_loss(pred, gt) + generalized_box_iou_loss(pred, gt, reduction='mean')
    else:
        # The mean of an empty tensor is NaN; this is 0 and keeps the graph connected
        box = out['box'].sum() * 0

    total = cls + box_weight * box
    return total, {'cls': cls.item(), 'box': box.item()}
