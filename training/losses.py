import torch.nn.functional as F


def compute_loss(outputs, labels):

    ce = F.cross_entropy(outputs["scores"], labels)

    collapse = outputs["collapse_loss"]

    loss = ce + 0.1 * collapse

    return loss
