from abc import ABC, abstractmethod
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter

import wandb


class Logger(ABC):
    @abstractmethod
    def __init__(self, name, **kwargs):
        pass

    @abstractmethod
    def log(self, data: dict, step=None):
        pass


class PrintLogger(Logger):
    def __init__(self, name, config=None, **kwargs):
        print(f"[INFO] Logger initialized. Running experiment {name}")
        print(f"[INFO] Current config: {config}")

    def log(self, data: dict, step=None):
        print(f"Step {step}")
        print("-" * 20)
        for key, value in data.items():
            print(f"{key}: {value}")
        print()


class WandbLogger(Logger):
    def __init__(self, name, config=None, model=None, **kwargs):
        """
        Initilizes wandb logger.

        [TODO] Fill params in wandb.init function.
        """

        wandb.init(
            project=...,
            entity=...,
            name=name,
            config=OmegaConf.to_container(config),
            mode="online",
            id=name,  # for resuming
        )

        if model:
            wandb.watch(model)

    def log(self, data: dict, step=None):
        wandb.log(data, step=step)


class TensorBoardLogger(Logger):
    def __init__(self, name, **kwargs):
        """
        Initilizes tensorboard logger.

        Only supports scalar logging.
        """

        self.writer = SummaryWriter(log_dir="logs/" + name)

    def log(self, data: dict, step=None):
        for key, value in data.items():
            self.writer.add_scalar(key, value, step)
