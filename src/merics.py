import torchmetrics
import torch
from typing import Any, Callable, Optional


class CustomMetric(torchmetrics.Metric):
    r"""
    Computes the following metric:

    math:: e = \frac{1}{m}\sum_{i=0}^{m-1}[\sum_{j=0}^{n-1}(\mathbf{x}_{ij}-\hat{\mathbf{x}}_{ij})^2]\cross
                [\sum_{j=0}^{n-1}\mathbf{x}_{ij}.\hat{\mathbf{x}}_{ij}]

    where :math:`\mathbf{x} \in \mathbb{R}^{m \cross n}` is the true vector of probabilities,
    :math:`\hat{\mathbf{x}} \in \mathbb{R}^{m \cross n}` is the predicted vector,
    :math:`m` is the number of sequences in the batch and :math:`n` is the number of classes

    Args:
        compute_on_step:
            Forward only calls ``update()`` and return None if this is set to False.
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step.
        process_group:
            Specify the process group on which synchronization is called.
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When `None`, DDP
            will be used to perform the allgather.

    """

    def __init__(
            self,
            compute_on_step: bool = True,
            dist_sync_on_step: bool = False,
            process_group: Optional[Any] = None,
            dist_sync_fn: Callable = None,
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state("total", default=torch.tensor(0, dtype=float), dist_reduce_fx="sum")
        self.add_state("n_observations", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """

        dist = torch.sum((target - preds) ** 2, dim=1)
        prod = torch.sum(torch.mul(target, preds), dim=1)

        self.total += torch.sum(torch.mul(dist, prod))
        self.n_observations += target.shape[0]

    def compute(self) -> torch.Tensor:
        return self.total / self.n_observations

