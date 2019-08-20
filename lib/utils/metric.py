"""
    hz_szh@zju.edu.cn
"""

from easydict import EasyDict


class AverageMeter(EasyDict):
    """Computes and stores the average and current value"""

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricRecorder(object):
    def __init__(self, targets=[]):
        self.targets = targets
        self._init_targets()

    def _init_targets(self):
        self.things = EasyDict()
        for t in self.targets:
            self.things.setdefault(t, AverageMeter())

    def update(self, **kwargs):
        for key, value in kwargs.items():
            try:
                self.things[key].update(value)
            except KeyError:
                print("Add keyword '{}' to targets".format(key))
                self.targets.append(key)
                self.things.setdefault(key, AverageMeter())
                self.things[key].update(value)

    def reset(self, targets=None):
        reset_items = self.targets if targets is None else targets
        for t in reset_items:
            self.things[t].reset()

    def getAvgVals(self, targets=None):
        value_dict = {}

        target_items = self.targets if targets is None else targets
        for t in target_items:
            value_dict.setdefault(t, self.things[t].avg)

        return value_dict


if __name__ == "__main__":
    targets = ['loss_seg', 'loss_flow', 'loss_random']
    loss_dict = {'loss_seg': 10, 'loss_flow': 2, 'loss_random': 5}
    loss_dict_2 = {'loss_new': 1}

    meter = MetricRecorder(targets=targets)
    meter.update(loss_dict)
    meter.update(loss_dict_2)
    meter.reset()
