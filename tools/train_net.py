import torch
import argparse
import os
import sys

sys.path.extend([".", ".."])

from lib.config import cfg
from lib.data import make_data_loader
from lib.solver import make_adam_optimizer, make_lr_scheduler
from lib.engine.inference import inference
from lib.engine.trainer import do_train
from lib.modeling.matcher import build_matching_model
from lib.utils.checkpoint import Checkpointer
from lib.utils.tensorboard import TensorBoard
from lib.utils.logger import make_getter


def train(cfg):
    
    
    getter = make_getter(cfg)
    model = build_matching_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model = model.to(device)
    model = torch.nn.DataParallel(model)

    optimizer = make_adam_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    # pass arguments to trainer, rather than cfg
    arguments = {}
    arguments["iteration"] = 0

    model_dir = os.path.join(cfg.MODEL_DIR, cfg.MODEL.NAME)
    checkpointer = Checkpointer(cfg, model, optimizer, scheduler, save_dir=model_dir)
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)

    tensorboard = TensorBoard(cfg) if cfg.TENSORBOARD.IS_ON else None

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        start_iter=arguments["iteration"],
    )

    checkpoint_period = cfg.TRAIN.CHECKPOINT_PERIOD

    do_train(
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
    )

    return model


def test(cfg, model):
    torch.cuda.empty_cache()
    dataset_name = cfg.DATASET.TEST
    model_dir = os.path.join(cfg.MODEL_DIR, cfg.MODEL.NAME)
    output_folder = os.path.join(model_dir, "inference", dataset_name)
    os.makedirs(output_folder, exist_ok=True)
    data_loader_val = make_data_loader(cfg, is_train=False)
    inference(
        cfg,
        model,
        data_loader_val,
        device=cfg.MODEL.DEVICE,
        output_folder=output_folder,
    )


def main():
    parser = argparse.ArgumentParser(description="Dense Correspondence")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="Path to config file",
        type=str,
    )
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        '--test_model',
        dest='test_model',
        action='store_true'
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = os.path.join(cfg.MODEL_DIR, cfg.MODEL.NAME)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if cfg.MODEL.TEST:
        # build model
        model = build_matching_model(cfg)
        model.to(cfg.MODEL.DEVICE)
        model = torch.nn.DataParallel(model)

        # load pretrained parameters
        model_dir = os.path.join(cfg.MODEL_DIR, cfg.MODEL.NAME)
        checkpointer = Checkpointer(cfg, model, save_dir=model_dir)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)

        test(cfg, model)
    else:
        model = train(cfg)

        if not args.skip_test:
            test(cfg, model)


if __name__ == "__main__":
    main()
