# coding: utf-8
"""Small helper utilities used by the PerformRecast pipeline."""
import torch


def dct2device(dct: dict, device) -> dict:
    """In-place: send every value of `dct` to `device`, wrapping non-tensors."""
    for key in dct:
        if isinstance(dct[key], torch.Tensor):
            dct[key] = dct[key].to(device)
        else:
            dct[key] = torch.tensor(dct[key]).to(device)
    return dct
