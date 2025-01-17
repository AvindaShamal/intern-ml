import torch
from cvas.model import NNModel


def test_forward():
    input_features = 3
    output_features = 10
    batch_size = 4
    image_size = 32

    dummy_input = torch.randn(
        batch_size, input_features, image_size, image_size
    )
    model = NNModel(input_features, output_features)
    output = model.forward(dummy_input)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, output_features)
