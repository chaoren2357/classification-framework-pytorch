import logging
import sys
import os
import shutil


class Logger(object):
    def __init__(self, resume = False) -> None:
        self.logger = logging.getLogger('logger')
        self.logger.setLevel(logging.DEBUG)
        self.has_basic_inform_logger = False
        self.resume = resume
        self.indexes = None
        self.dict = {}
        self.basic_inform_handler = logging.FileHandler("basicInformation_temp.log")
        self.training_inform_handler = None
        self.stream_handler = logging.StreamHandler(sys.stderr)
        self.stream_handler.setLevel(logging.DEBUG)
        self.stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

        self.logger.addHandler(self.stream_handler)

    def prepare_dataset(self, dataset_name: str, num_classes: int) -> None:
        if self.resume:
            pass
        else:
            self.basic_inform_handler.setLevel(logging.INFO)
            self.basic_inform_handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(self.basic_inform_handler)
            self.has_basic_inform_logger = True
        self.logger.debug('==> Preparing dataset')
        self.logger.info('Dataset: %s' % dataset_name)
        self.logger.info('Number of classes: %s' % num_classes)

    def define_model(self, arch, params, criterion, optim, lr, momentum, weight_decay) -> None:
        self.logger.debug("==> Creating model")
        self.logger.info("Model: %s" % arch)
        self.logger.info('Total params: %.2fM' % (sum(p.numel() for p in params) / 1000000.0))
        self.logger.info("Criterion: %s", criterion.__class__.__name__)
        self.logger.info("Optimizer: %s", optim.__class__.__name__)
        self.logger.info("Learning rate: %s" % lr)
        self.logger.info("Momentum: %s" % momentum)
        self.logger.info("Weight decay: %s" % weight_decay)

    def choose_device(self, use_cuda, devices=None) -> None:
        if use_cuda:
            self.logger.info('Device: GPU: ', devices)
        else:
            self.logger.info('Device: CPU')

    def ready_training(self, path, indexes=None) -> None:
        ## End basic information setting
        if indexes is None:
            self.indexes = ['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.']
        else:
            self.indexes = indexes
        if not os.path.isdir(path):
            assert "You can't set this path!"
        if not self.has_basic_inform_logger:
            assert "You don't have the basic information file!"
        shutil.move("basicInformation_temp.log", os.path.join(path, "basicInformation.log"))
        self.logger.removeHandler(self.basic_inform_handler)
        self.logger.removeHandler(self.stream_handler)

        ## Start logging
        self.training_inform_handler = logging.FileHandler(os.path.join(path, "trainingInformation.log"))
        self.logger.addHandler(self.training_inform_handler)
        first_row = ''
        for i, index in enumerate(indexes):
            first_row = first_row + index.ljust(15)
            self.dict[index] = []
            self.dict[i] = index
            # dict has two things: Indexes with corresponding list, and Order number with corresponding index.
            # e.g. dict = {"Accuracy":[0.5,0.6], "Epoch":[0, 1], 0:"Epoch", 1:Accuracy"}
        self.logger.info(first_row)


    def resume_training(self, ckpt):
        self.logger.info('==> Resuming from checkpoint..')
        self.logger.info("    Resuming from %s" % ckpt)

    def append(self, numbers):
        assert len(self.indexes) == len(numbers), 'Numbers do not match indexes'
        row = ''
        for i, num in enumerate(numbers):
            self.dict[self.dict[i]].append(num)
            row = row + str(round(num,3)).ljust(15)
        self.logger.info(row)


