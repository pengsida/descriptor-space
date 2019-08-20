from .coco import COCODataset
from .coco_v1 import COCODatasetV1
from .coco_v2 import COCODatasetV2
from .coco_v3 import COCODatasetV3
from .coco_v4 import COCODatasetV4
from .coco_v5 import COCODatasetV5
from .coco_rotation import COCODatasetRotation
from .coco_hpatches import COCODatasetHpatches
from .coco_perspective import COCOPerspective
from .hpatches import HpatchesDataset
from .hpatches import HpatchesViewpoint
from .hpatches import HpatchesIllum
from .hpatches_v1 import HpatchesV1
from .hpatches_v2 import HpatchesV2
from .hpatches_rotation import HpatchesRotation

__all__ = ["COCODataset", "HpatchesDataset",
           "HpatchesViewpoint", "HpatchesIllum", "HpatchesV1",
           "COCODatasetV1", "COCODatasetV2", "HpatchesV2", 'COCODatasetRotation',
           'HpatchesRotation', 'COCODatasetV3', "COCODatasetHpatches",
           "COCODatasetV4", "COCODatasetV5", "COCOPerspective"]
