import time
import torch
import datetime

from lib.utils.metric_logger import MetricLogger
from lib.utils.misc import unnormalize
from lib.utils.logger import logger
from .inference import test


def reduce_loss_dict(loss_dict):
    reduced_losses = {k: torch.mean(v) for k, v in loss_dict.items()}
    return reduced_losses
    # return loss_dict['loss']


def do_train(
    cfg,
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    tensorboard,
    device,
    checkpoint_period,
    arguments,
    getter
):
    print("Start training")
    
    
    # keep training statistics
    meters = MetricLogger(delimiter="  ")

    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    for iteration, (images0, images1, targets, _) in enumerate(data_loader, start_iter):
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        images0 = images0.to(device)
        images1 = images1.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        targets.update(dict(iteration=iteration))

        targets["iteration"] = iteration
        loss_dict, result_dict = model(images0, images1, targets)
        loss_dict = reduce_loss_dict(loss_dict)
        meters.update(**loss_dict)
        
        loss = model.module.get_loss_for_bp(loss_dict)
        # losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step(loss.cpu())

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)
        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == (max_iter - 1):
            print(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
            
            tb_data = getter.get_tensorboard_data()
            if tensorboard is not None:
                metric_dict = meters.state_dict()
                tensorboard.update(**metric_dict)
                tensorboard.update(**tb_data)
                tensorboard.add('train', iteration)

        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    print(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter - start_iter + 1e-5)
        )
    )
