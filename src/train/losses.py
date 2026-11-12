import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import generalized_box_iou_loss

BOX_TERMS = {'l1', 'giou'}


def detection_loss(
    out: dict[str, Tensor],
    batch: dict[str, Tensor],
    pos_weight: float,
    box_weight: float = 1.0,
    box_terms: tuple[str, ...] = ('l1', 'giou'),
) -> tuple[Tensor, dict[str, float]]:
    """
    Classification loss on every slice plus box loss on the positive ones.

    Args:
        out: Output of SliceNet, with 'logit' (B,) and 'box' (B, 4).
        batch: Batch from SliceDataset, already on the model's device.
        pos_weight: Weight of the positive class in the BCE.
        box_weight: Weight of the box loss in the total.
        box_terms: Which box losses to add: any non-empty subset of
            ('l1', 'giou'). Both by default; one alone is an ablation.

    Returns:
        The total loss, and its 'cls' and 'box' terms as floats for logging.
    """
    unknown = set(box_terms) - BOX_TERMS
    if not box_terms or unknown:
        raise ValueError(f"box_terms must be a non-empty subset of {sorted(BOX_TERMS)}, got {box_terms}")

    # Created on the logit's device: a CPU tensor here fails on MPS
    cls = F.binary_cross_entropy_with_logits(
        out['logit'],
        batch['label'],
        pos_weight=torch.tensor(pos_weight, device=out['logit'].device),
    )

    mask = batch['has_box']
    if mask.any():
        pred, gt = out['box'][mask], batch['box'][mask]
        box = 0
        if 'l1' in box_terms:
            box = box + F.l1_loss(pred, gt)
        if 'giou' in box_terms:
            box = box + generalized_box_iou_loss(pred, gt, reduction='mean')
    else:
        # The mean of an empty tensor is NaN; this is 0 and keeps the graph connected
        box = out['box'].sum() * 0

    total = cls + box_weight * box
    return total, {'cls': cls.item(), 'box': box.item()}
