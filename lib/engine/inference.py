# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import tempfile
import time
import os
from collections import OrderedDict

import torch
from tqdm import tqdm

from lib.data.datasets.evaluation import make_evaluator
from lib.data import make_data_loader


def evaluate(model, data_loader, device):
    model.eval()
    cpu_device = torch.device("cpu")
    dataset = data_loader.dataset
    evaluator = make_evaluator(dataset, evaluate.__name__)

    for i, data in tqdm(enumerate(data_loader)):
        images0 = data[0].to(device)
        images1 = data[1].to(device)
        targets = data[2]
        indexes = data[-1]

        with torch.no_grad():
            left_scales = targets['left_scale'].cuda() if 'left_scale' in targets else 2
            scales = targets['scale'].cuda()
            descriptors0 = model.module.inference(images0, left_scales, 'left').to(cpu_device)
            descriptors1 = model.module.inference(images1, scales, 'right').to(cpu_device)

        for bi in range(images0.shape[0]):
            evaluator.evaluate(descriptors0[bi], descriptors1[bi], indexes[bi], targets['H'][bi])

        evaluator.average_precision()
    evaluator.wrong_pixel_scale()


def evaluate_desc_dict(model, data_loader, device):
    model.eval()
    cpu_device = torch.device("cpu")
    dataset = data_loader.dataset
    evaluator = make_evaluator(dataset, evaluate_desc_dict.__name__)

    for i, data in tqdm(enumerate(data_loader)):
        images0 = data[0].to(device)
        images1 = data[1].to(device)
        targets = data[2]
        indexes = data[-1]

        with torch.no_grad():
            scales0 = targets['left_scale'].cuda() if 'left_scale' in targets else 2
            scales1 = targets['scale'].cuda()
            descs0 = {k: v.to(cpu_device) for k, v in model.module.inference(images0, scales0, 'left').items()}
            descs1 = {k: v.to(cpu_device) for k, v in model.module.inference(images1, scales1, 'right').items()}

        for bi in range(images0.shape[0]):
            desc0 = {k: v[bi] for k, v in descs0.items()}
            desc1 = {k: v[bi] for k, v in descs1.items()}
            evaluator.evaluate(desc0, desc1, indexes[bi], targets['H'][bi])

        evaluator.average_precision()
    evaluator.wrong_pixel_scale()


def evaluate_scale(model, data_loader, device):
    model.eval()
    cpu_device = torch.device("cpu")
    dataset = data_loader.dataset
    evaluator = make_evaluator(dataset, evaluate_scale.__name__)

    for i, data in tqdm(enumerate(data_loader)):
        images0 = data[0].to(device)
        images1 = data[1].to(device)
        left_scales = data[2]['left_scale']
        scales = data[2]['scale']
        msk = data[2]['msk']
        indexes = data[-1]

        with torch.no_grad():
            scales_pred = model.module.inference(images0, images1, left_scales.cuda(), scales.cuda()).to(cpu_device)

        for bi in range(images0.shape[0]):
            evaluator.evaluate(scales_pred[bi], scales[bi], images0[bi], images1[bi], indexes[bi], msk[bi])
        evaluator.average_precision()


def evaluate_scale_desc(model, data_loader, device):
    model.eval()
    cpu_device = torch.device("cpu")
    dataset = data_loader.dataset
    evaluator = make_evaluator(dataset, evaluate_scale_desc.__name__)

    for i, data in tqdm(enumerate(data_loader)):
        images0 = data[0].to(device)
        images1 = data[1].to(device)
        left_scales = data[2]['left_scale'].to(device)
        scales = data[2]['scale'].to(device)
        msk = data[2]['msk']
        target = data[2]
        indexes = data[-1]

        with torch.no_grad():
            descriptors0, descriptors1, scales_pred = model.module.inference(images0, images1, left_scales, scales)
            descriptors0 = descriptors0.to(cpu_device)
            descriptors1 = descriptors1.to(cpu_device)
            scales_pred = scales_pred.to(cpu_device)
            scales = scales.to(cpu_device)

        for bi in range(images0.shape[0]):
            evaluator.evaluate(
                descriptors0[bi], descriptors1[bi], indexes[bi], target['H'][bi],
                scales_pred[bi], scales[bi],
                images0[bi], images1[bi], msk[bi]
            )
        evaluator.average_precision()


def inference(
    cfg,
    model,
    data_loader,
    device="cuda",
    expected_results=(),
    output_folder=None,
    **kwargs,
):

    device = torch.device(device)
    num_devices = torch.cuda.device_count()

    logger = logging.getLogger("lib.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} images".format(len(dataset)))
    start_time = time.time()
    # evaluate_scale_desc(model, data_loader, device)
    # evaluate_scale(model, data_loader, device)
    # evaluate_desc_dict(model, data_loader, device)
    evaluate(model, data_loader, device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    logger.info(
        "Total inference time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )


def test(cfg, model, **kwargs):
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
        **kwargs,
    )
