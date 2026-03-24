import torch
import torch.nn.functional as F


def get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))


def get_available_activation_fns():
    return [
        "relu",
        "gelu",
        "tanh",
        "linear",
    ]
