from omegaconf import DictConfig

import hydra
import torch as th

import utils
import logger as _logger
import models


class Tester:
    def __init__(self, model, dataloader, logger, name, *, device=None):
        self.model = model
        self.dataloader = dataloader
        self.logger = logger
        self.name = name

        if not device:
            device = "cuda" if th.cuda.is_available() else "cpu"
        self.device = th.device(device)

        print(f"""[INFO] Using device: {self.device}""")

        self.model = self.model.to(self.device)

    def test(self):
        pass


@hydra.main(version_base=None, config_path="exp", config_name="test")
def test(config: DictConfig) -> None:
    # get run name
    test_name = utils.get_test_name()

    # initialize logger
    logger = getattr(_logger, config.logger)

    # initialize model
    model = getattr(models, config.exp.model)(**config.exp.model_params)

    # load ckpt
    model.load_state_dict(th.load(config.ckpt)["model"])

    # initialize dataloader
    dataloader = th.utils.data.DataLoader(
        utils.get_dataset(config.test_path), **config.dataloader_params
    )

    # initialize Tester
    tester = Tester(
        model=model,
        dataloader=dataloader,
        logger=logger,
        name=test_name,
        **config.tester_params,
    )

    # test
    tester.test()


if __name__ == "__main__":
    test()
