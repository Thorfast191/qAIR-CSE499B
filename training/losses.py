import torch
import torch.nn.functional as F


def compute_loss(outputs, labels):

    scores = outputs["scores"]

    ce = F.cross_entropy(scores, labels)

    collapse = outputs["collapse_loss"]

    loss = ce + 0.01 * collapse

    return loss