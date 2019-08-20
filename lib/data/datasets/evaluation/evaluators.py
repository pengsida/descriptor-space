from lib.data import datasets

from .hpatches import HpatchesEvaluator, HpatchesDescDictEvaluator
from .coco import COCOEvaluator, COCOScaleEvaluator, COCOScaleDescEvaluator


def make_evaluator(dataset, inference_name):
    evaluators = {
        'evaluate': {
            datasets.HpatchesDataset: HpatchesEvaluator,
            datasets.HpatchesViewpoint: HpatchesEvaluator,
            datasets.COCODataset: COCOEvaluator
        },
        'evaluate_desc_dict': {
            datasets.HpatchesViewpoint: HpatchesDescDictEvaluator
        },
        'evaluate_scale': {
            datasets.HpatchesDataset: COCOScaleEvaluator,
            datasets.HpatchesViewpoint: COCOScaleEvaluator,
            datasets.HpatchesV2: COCOScaleEvaluator,
            datasets.COCODatasetHpatches: COCOScaleEvaluator
        },
        'evaluate_scale_desc': {
            datasets.HpatchesDataset: COCOScaleDescEvaluator,
            datasets.COCODatasetHpatches: COCOScaleDescEvaluator,
            datasets.HpatchesViewpoint: COCOScaleDescEvaluator
        }
    }
    return evaluators[inference_name][dataset.__class__](dataset)
