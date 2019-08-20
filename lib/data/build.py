from lib.data.transforms import build_transforms
from . import datasets as D
from . import samplers
import torch
import torch.utils.data
from lib.utils.imports import import_file


def build_dataset(cfg, dataset_name, transforms, dataset_catalog, is_train=True):
    data = dataset_catalog.get(dataset_name, cfg)
    factory = getattr(D, data["factory"])
    args = data["args"]
    args["cfg"] = cfg
    args["transforms"] = transforms
    dataset = factory(**args)
    return dataset


def make_data_sampler(dataset, shuffle):
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_batch_data_sampler(dataset, sampler, images_per_batch, num_iters=None, start_iter=0):
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, images_per_batch, drop_last=False
    )
    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler


def make_data_loader(cfg, is_train=True, start_iter=0, max_iter=None):
    num_gpus = int(torch.cuda.device_count())
    if is_train:
        images_per_batch = cfg.TRAIN.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "TRAIN.IMS_PER_BATCH ({}) must be divisible by the number "
        "of GPUs ({}) used.".format(images_per_batch, num_gpus)
        shuffle = True
        num_iters = cfg.TRAIN.MAX_ITER if max_iter is None else max_iter
    else:
        images_per_batch = cfg.TEST.IMS_PER_BATCH
        assert (
            images_per_batch % num_gpus == 0
        ), "TEST.IMS_PER_BATCH ({}) must be divisible by the number "
        "of GPUs ({}) used.".format(images_per_batch, num_gpus)
        shuffle = False
        num_iters = None
        start_iter = 0

    paths_catalog = import_file(
        'lib.config.path', cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog
    dataset_name = cfg.DATASET.TRAIN if is_train else cfg.DATASET.TEST

    transforms = build_transforms(cfg, is_train)
    dataset = build_dataset(cfg, dataset_name, transforms, DatasetCatalog, is_train)
    sampler = make_data_sampler(dataset, shuffle)
    batch_sampler = make_batch_data_sampler(dataset, sampler, images_per_batch, num_iters, start_iter)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers
    )

    return data_loader
