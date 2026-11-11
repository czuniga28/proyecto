import pytest
import torch

from src.models.net import SliceNet, cxcywh_to_xyxy


def test_cxcywh_to_xyxy():
    boxes = torch.tensor([[0.5, 0.4, 0.2, 0.6]])
    expected = torch.tensor([[0.4, 0.1, 0.6, 0.7]])
    assert torch.allclose(cxcywh_to_xyxy(boxes), expected)


def test_cxcywh_to_xyxy_keeps_leading_dims():
    assert cxcywh_to_xyxy(torch.rand(2, 3, 4)).shape == (2, 3, 4)


@pytest.fixture(scope='module')
def net():
    return SliceNet(pretrained=False).eval()


@pytest.mark.parametrize('batch', [1, 3])
def test_forward_shapes(net, batch):
    with torch.no_grad():
        out = net(torch.randn(batch, 3, 224, 224))
    assert out['logit'].shape == (batch,)
    assert out['box'].shape == (batch, 4)


def test_boxes_are_valid(net):
    with torch.no_grad():
        box = net(torch.randn(8, 3, 224, 224))['box']
    assert (box[:, 2] >= box[:, 0]).all()
    assert (box[:, 3] >= box[:, 1]).all()
    # Width and height come from a sigmoid, so they never exceed 1
    assert ((box[:, 2:] - box[:, :2]) <= 1).all()


def test_backbone_outputs_features(net):
    with torch.no_grad():
        assert net.backbone(torch.randn(2, 3, 224, 224)).shape == (2, 512)
