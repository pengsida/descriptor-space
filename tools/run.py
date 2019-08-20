import argparse
import sys
sys.path.extend([".", ".."])

from lib.config import cfg


def run_dataset(cfg):
    from lib.data.build import make_data_loader
    from tqdm import tqdm
    data_loader = make_data_loader(cfg, is_train=True)
    for data in tqdm(data_loader.dataset):
        pass


def run_model(cfg):
    from lib.modeling.matcher import build_matching_model
    import torch
    model = build_matching_model(cfg).cuda()
    images = torch.randn([1, 6, 600, 800]).cuda()
    model.training = False
    model(images)


def run_match(cfg):
    import torch
    from lib.utils.nn_set2set_match.nn_set2set_match_layer import nn_set2set_match_cuda, nn_set2set_match_numpy
    import numpy as np
    import time

    for _ in range(100):
        descs0 = np.random.rand(3000, 2, 128)
        descs1 = np.random.rand(3000, 2, 128)

        now = time.time()
        idxs_cuda = nn_set2set_match_cuda(torch.tensor(descs0).unsqueeze(0).cuda(), torch.tensor(descs1).unsqueeze(0).cuda())
        print(time.time() - now)


if __name__ == "__main__":
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
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER
    )
    parser.add_argument(
        "--type",
        type=str,
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    globals()["run_" + args.type](cfg)
