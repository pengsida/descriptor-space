# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import torch
from lib.utils.model_serialization import load_state_dict
from lib.utils.base import read_pickle, save_pickle


class Checkpointer(object):
    def __init__(
        self,
        cfg,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
    ):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        data = {}
        data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        print("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        self.update_checkpoint(save_file)

    def load(self, f=None):
        if not self.cfg.TRAIN.RESUME:
            print("cfg.TRAIN.RESUME is set False. No model will be loaded")
            return {}
        if self.has_checkpoint():
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()
        if not f:
            # no checkpoint could be found
            print("No checkpoint found. Initializing model from scratch")
            return {}
        print("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint)
        if "optimizer" in checkpoint and self.optimizer:
            print("Loading optimizer from {}".format(f))
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if "scheduler" in checkpoint and self.scheduler:
            print("Loading scheduler from {}".format(f))
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        # return any further checkpoint data
        return checkpoint

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "checkpoint.pkl")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "checkpoint.pkl")
        try:
            checkpoints = read_pickle(save_file)
            last_saved = os.path.join(self.save_dir, checkpoints[-1])
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def update_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "checkpoint.pkl")
        if os.path.exists(save_file):
            checkpoints = read_pickle(save_file)
        else:
            checkpoints = []
        checkpoints.append(os.path.basename(last_filename))
        if len(checkpoints) > self.cfg.TRAIN.NUM_CHECKPOINT:
            checkpoint_name = checkpoints.pop(0)
            checkpoint = os.path.join(self.save_dir, checkpoint_name)
            if os.path.exists(checkpoint) and checkpoint_name not in checkpoints:
                os.remove(checkpoint)
        save_pickle(checkpoints, save_file)

    def _load_file(self, f):
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint):
        load_state_dict(self.model, checkpoint.pop("model"))
