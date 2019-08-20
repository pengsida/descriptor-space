import os


class DatasetCatalog(object):
    DATA_DIR = "data"
    DATASETS = {
        "COCO_PERSPECTIVE": (
            "MSCOCO2017",
        ),
        "COCO_V5": (
            "MSCOCO2017",
        ),
        "COCO_HPATCHES": (
            "MSCOCO2017",
        ),
        "COCO_V3": (
            "MSCOCO2017",
        ),
        "COCO_V4": (
            "MSCOCO2017",
        ),
        "HPATCHES_ROTATION": (
            "HPATCHES",
        ),
        "COCO_ROTATION": (
            "MSCOCO2017",
        ),
        "HPATCHES_V2": (
            "HPATCHES",
        ),
        "COCO_V2": (
            "MSCOCO2017",
        ),
        "COCO_V1": (
            "MSCOCO2017",
        ),
        "HPATCHES_V1": (
            "HPATCHES/",
        ),
        "HPATCHES_VIEWPOINT": (
            "HPATCHES/",
        ),
        "HPATCHES_ILLUM": (
            "HPATCHES/",
        ),
        "HPATCHES": (
            "HPATCHES/",
        ),
        "COCO": (
            "MSCOCO2017/",
        ),
        "ShapeNet_car_mini-large_val": (
            "ShapeNet/renders/02958343",
            "ShapeNet/renders/02958343/ShapeNet_car_mini-large_val.pkl",
        ),
    }

    @staticmethod
    def get(name, cfg=None):
        data_dir = DatasetCatalog.DATA_DIR
        attrs = DatasetCatalog.DATASETS[name]
        if "ShapeNet" in name:
            args = dict(
                ann_file=os.path.join(data_dir, attrs[1]),
            )
            return dict(
                factory="ShapeNetDataset",
                args=args,
            )
        elif "COCO" in name:
            args = dict(
                root=os.path.join(data_dir, attrs[0]),
            )
            classname = {
                "COCO": "COCODataset",
                "COCO_V1": "COCODatasetV1",
                "COCO_V2": "COCODatasetV2",
                "COCO_V3": "COCODatasetV3",
                "COCO_V4": "COCODatasetV4",
                "COCO_V5": "COCODatasetV5",
                "COCO_ROTATION": "COCODatasetRotation",
                "COCO_HPATCHES": "COCODatasetHpatches",
                "COCO_PERSPECTIVE": "COCOPerspective"
            }[name]
            if name == 'COCO_ROTATION':
                args['angle'] = cfg.DATASET.COCO_ANGLE
            return dict(
                factory=classname,
                args=args,
            )
        elif "HPATCHES" in name:
            args = dict(
                root=os.path.join(data_dir, attrs[0]),
            )
            classname = {
                "HPATCHES": "HpatchesDataset",
                "HPATCHES_VIEWPOINT": "HpatchesViewpoint",
                "HPATCHES_ILLUM": "HpatchesIllum",
                "HPATCHES_V1": "HpatchesV1",
                "HPATCHES_V2": "HpatchesV2",
                "HPATCHES_ROTATION": "HpatchesRotation"}[name]

            if name == 'HPATCHES_ROTATION':
                args['angle'] = cfg.DATASET.HPATCHES_ANGLE
            return dict(
                factory=classname,
                args=args,
            )

        raise RuntimeError("Dataset: {} not available".format(name))
