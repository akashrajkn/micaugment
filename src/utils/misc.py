import torch


def smoothmax(a, b):

    exp_a       = torch.exp(a)
    exp_b       = torch.exp(b)
    numerator   = a * exp_a + b * exp_b
    denominator = exp_a + exp_b

    return numerator / denominator


def smoothmin(a, b):
    exp_m_a     = torch.exp(-a)
    exp_m_b     = torch.exp(-b)
    numerator   = a * exp_m_a + b * exp_m_b
    denominator = exp_m_a + exp_m_b

    return numerator / denominator
