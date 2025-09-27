
""" Расчёт метрик """


import torch
import torch.nn.functional as F


def compute_metrics(p):
    """Расчет метрики потери."""
    preds, labels = p.predictions, p.label_ids
    loss = float(F.cross_entropy(
        torch.tensor(preds),
        torch.tensor(labels))
    )
    return {"loss": loss}
