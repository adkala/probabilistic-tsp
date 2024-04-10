from omegaconf import DictConfig
from tqdm.rich import tqdm

import hydra
import torch as th
import os

import utils
import logger as _logger
import models


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        dataloader,
        logger,
        name,
        *,
        num_epochs,
        log_interval,
        save_interval,
        val_dataloader=None,
        device=None,
        start_epoch=0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.logger = logger
        self.name = name

        self.num_epochs = num_epochs
        self.log_interval = log_interval
        self.save_interval = save_interval

        self.val_dataloader = val_dataloader

        self.epoch = start_epoch

        if not device:
            device = "cuda" if th.cuda.is_available() else "cpu"
        self.device = th.device(device)

        print(f"""[INFO] Using device: {self.device}""")

        self.model = self.model.to(self.device)

        self.loss_fn = th.nn.CrossEntropyLoss()

    def train(self):
        with tqdm(total=self.num_epochs * len(self.dataloader)) as pbar:
            pbar.n = self.epoch * len(self.dataloader)
            pbar.refresh()
            for epoch in range(self.epoch, self.num_epochs):
                self.epoch = epoch
                loss = self.step(pbar)

                if epoch % self.log_interval == 0:
                    self.logger.log({"loss": loss}, step=epoch)
                    self.validate()

                if epoch % self.save_interval == 0:
                    if not os.path.exists(f"models/{self.name}"):
                        os.makedirs(f"models/{self.name}")

                    th.save(
                        {
                            "model": self.model.state_dict(),
                            "optim": self.optimizer.state_dict(),
                            "epoch": self.epoch,
                            "name": self.name,
                        },
                        f"models/{self.name}/e{self.epoch}.pt",
                    )

    def step(self, pbar=None):
        total_loss, steps = 0, 0
        for x, y in self.dataloader:
            x, y = x.to(self.device), y.to(self.device)

            y_pred = self.model(x)

            loss = self.loss_fn(y_pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            steps += 1

            if pbar:
                pbar.update(1)
                pbar.set_description(f"Epoch {self.epoch} | Loss: {loss.item()}")
        return total_loss / steps

    def validate(self):  # implement for validation, default is None
        """
        [TODO] Implement for validation, default is None.

        Should be called every log_interval. Log values to logger.
        """
        pass


@hydra.main(version_base=None, config_path="exp", config_name="train")
def train(config: DictConfig) -> None:
    exp_cfg = config.exp

    # initialize model
    model = getattr(models, exp_cfg.model)(**exp_cfg.model_params)

    # initialize optimizer
    optim = getattr(th.optim, exp_cfg.optim)(model.parameters(), **exp_cfg.optim_params)

    # get run name
    run_name = utils.get_run_name(exp_cfg)

    # check for checkpoint
    start_epoch = 0
    if "ckpt" in config:
        ckpt = th.load(str(config.ckpt))
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])

        start_epoch = ckpt["epoch"] + 1
        run_name = ckpt["name"]

    # initialize dataloader
    dataloader = th.utils.data.DataLoader(
        utils.get_dataset(config.data_path), **exp_cfg.dataloader_params
    )

    # initialize logger
    logger = getattr(_logger, config.logger)(run_name, config=exp_cfg, model=model)

    # initialize Trainer
    trainer = Trainer(
        model=model,
        optimizer=optim,
        dataloader=dataloader,
        logger=logger,
        name=run_name,
        **exp_cfg.trainer_params,
        start_epoch=start_epoch,
    )

    trainer.train()


if __name__ == "__main__":
    train()
