import torch
from src.merics import CustomMetric


class TestMetric(object):

    # test basic calculation of the metric
    def test_custom_metric_calculation(self):
        custom_metric = CustomMetric()
        preds = torch.tensor([[0.9, 0.4], [0.1, 0.8]])  # batch_size=2, n=2
        target = torch.tensor([[1., 0.], [0., 1.]])
        expected = torch.tensor((((0.9-1.)**2 + (0.4-0.)**2) * (0.9*1. + 0.4*0.) + ((0.1-0.)**2 + (0.8-1.)**2) *
                                 (0.1*0. + 0.8*1.)) / 2, dtype=torch.float64)  # =0.0965
        assert torch.allclose(custom_metric(preds, target), expected)
