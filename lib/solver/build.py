import torch


def make_adam_optimizer(cfg, model: torch.nn.Module):
    params = []
    lr = cfg.TRAIN.BASE_LR
    weight_decay = cfg.TRAIN.WEIGHT_DECAY

    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    optimizer = torch.optim.Adam(params, lr, weight_decay=weight_decay)

    return optimizer
