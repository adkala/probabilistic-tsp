from omegaconf import DictConfig
from functools import reduce

import time

RUN_NAME_DICT = {
    "model": "",
    "optim": "",
    "optim_params.lr": "lr",  # multi level key
}


def get_run_name(config: DictConfig):
    run_name = ""
    for key, value in RUN_NAME_DICT.items():
        run_name += f"{value}{reduce(lambda x, y: x[y], key.split('.'), config)}_"
    run_name += f"{time.strftime('%Y-%m-%d-%H-%M-%S')}"
    return run_name


def get_test_name(config: DictConfig):
    return f"test_{time.strftime('%Y-%m-%d-%H-%M-%S')}_{config.ckpt}"


def get_dataset(filepath, flatten=False):
    """
    :param filepath: Path to the dataset.
    :return: Dataset object.

    [TODO] Replace this function.
    """
    pass
