import math

import pytest
import torch
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou_loss

from src.models.net import SliceNet
from src.train.losses import detection_loss

GT_BOX = [0.2, 0.3, 0.6, 0.7]


def make_batch(labels):
    has_box = torch.tensor(labels, dtype=torch.bool)
    box = torch.zeros(len(labels), 4)
    box[has_box] = torch.tensor(GT_BOX)
    return {'label': has_box.float(), 'box': box, 'has_box': has_box}


def make_out(logits, boxes):
    return {
        'logit': torch.tensor(logits, requires_grad=True),
        'box': torch.tensor(boxes, requires_grad=True),
    }


def test_no_positives_gives_zero_box_loss_and_gradient():
    out = make_out([0.1, -0.3], [[0.1, 0.1, 0.5, 0.5]] * 2)
    total, parts = detection_loss(out, make_batch([False, False]), pos_weight=2.0)
    total.backward()

    assert parts['box'] == 0.0
    assert math.isfinite(total.item())
    # Graph stays connected through the box head, with zero gradient
    assert torch.equal(out['box'].grad, torch.zeros(2, 4))


def test_perfect_box_gives_zero_box_loss():
    out = make_out([3.0, -3.0], [GT_BOX, [0.0, 0.0, 0.1, 0.1]])
    _, parts = detection_loss(out, make_batch([True, False]), pos_weight=1.0)
    assert parts['box'] == pytest.approx(0.0, abs=1e-6)


def test_negative_boxes_are_ignored():
    batch = make_batch([True, False])
    a = make_out([0.0, 0.0], [GT_BOX, [0.0, 0.0, 0.1, 0.1]])
    b = make_out([0.0, 0.0], [GT_BOX, [0.5, 0.5, 0.9, 0.9]])
    _, parts_a = detection_loss(a, batch, pos_weight=1.0)
    _, parts_b = detection_loss(b, batch, pos_weight=1.0)
    assert parts_a['box'] == parts_b['box']


def test_pos_weight_scales_positive_term():
    batch = make_batch([True, True])
    _, parts_1 = detection_loss(make_out([0.7, -1.2], [GT_BOX] * 2), batch, pos_weight=1.0)
    _, parts_2 = detection_loss(make_out([0.7, -1.2], [GT_BOX] * 2), batch, pos_weight=2.0)
    assert parts_2['cls'] == pytest.approx(2 * parts_1['cls'], rel=1e-5)


def test_total_combines_terms():
    out = make_out([0.5, -1.0], [[0.1, 0.2, 0.5, 0.9], [0.0, 0.0, 0.1, 0.1]])
    total, parts = detection_loss(out, make_batch([True, False]), pos_weight=2.0, box_weight=4.0)
    assert total.item() == pytest.approx(parts['cls'] + 4.0 * parts['box'], rel=1e-5)


def test_gradient_reaches_every_part():
    torch.manual_seed(0)
    net = SliceNet(pretrained=False).train()
    out = net(torch.randn(4, 3, 64, 64))
    total, _ = detection_loss(out, make_batch([True, False, True, False]), pos_weight=2.0)
    total.backward()

    params = {
        'cls_head': net.cls_head[1].weight,
        'box_head': net.box_head[1].weight,
        'backbone.conv1': net.backbone.conv1.weight,
    }
    for name, p in params.items():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_box_terms_select_each_loss():
    batch = make_batch([True, False])
    pred = [[0.1, 0.2, 0.5, 0.9], [0.0, 0.0, 0.1, 0.1]]
    l1 = F.l1_loss(torch.tensor(pred[:1]), torch.tensor([GT_BOX])).item()
    giou = generalized_box_iou_loss(torch.tensor(pred[:1]), torch.tensor([GT_BOX]), reduction='mean').item()

    def box(terms):
        return detection_loss(make_out([0.0, 0.0], pred), batch, 1.0, box_terms=terms)[1]['box']

    assert box(('l1',)) == pytest.approx(l1)
    assert box(('giou',)) == pytest.approx(giou)
    assert box(('l1', 'giou')) == pytest.approx(l1 + giou)


@pytest.mark.parametrize('terms', [(), ('l2',)])
def test_box_terms_rejects_unknown_or_empty(terms):
    with pytest.raises(ValueError, match='box_terms'):
        detection_loss(make_out([0.0], [GT_BOX]), make_batch([True]), 1.0, box_terms=terms)
