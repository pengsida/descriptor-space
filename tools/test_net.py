import argparse
import os
import sys

sys.path.extend([".", ".."])

import torch
from lib.config import cfg
from lib.data import make_data_loader
from lib.engine.inference import inference
from lib.modeling.matcher import build_matching_model
from lib.utils.checkpoint import Checkpointer


def main():
    parser = argparse.ArgumentParser(description="Dense Correspondence")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    model = build_matching_model(cfg)
    model.to(cfg.MODEL.DEVICE)
    model = torch.nn.DataParallel(model)

    model_dir = os.path.join(cfg.MODEL_DIR, cfg.MODEL.NAME)
    checkpointer = Checkpointer(cfg, model, save_dir=model_dir)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)

    dataset_name = cfg.DATASET.TEST
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


if __name__ == "__main__":
    main()
