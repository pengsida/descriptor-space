from tensorboardX import SummaryWriter
from easydict import EasyDict
import os


class TensorBoard(object):
    def __init__(self, cfg):
        self.cfg = cfg
        logidr = os.path.join(cfg.TENSORBOARD.LOG_DIR, cfg.MODEL.NAME)
        if os.path.exists(logidr) and not cfg.TRAIN.RESUME:
            os.system('rm -r {}'.format(logidr))
        self.writer = SummaryWriter(log_dir=logidr)
        self.things = EasyDict()

    def update(self, **kwargs):
        for key, value in kwargs.items():
            self.things[key] = value

    def add(self, prefix, global_step):
        pattern = '/'.join([prefix, '{}'])
        # pattern = '{}'
        
        targets = self.cfg.TENSORBOARD.TARGETS
        scalars = targets.SCALAR
        images = targets.IMAGE
        for scalar in scalars:
            if scalar not in self.things:
                continue
            self.writer.add_scalar(pattern.format(scalar), self.things[scalar], global_step)
            
        for image in images:
            if image in self.things:
                self.writer.add_image(pattern.format(image), self.things[image], global_step)
