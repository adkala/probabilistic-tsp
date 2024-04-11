from omegaconf import DictConfig
from tqdm.rich import tqdm

from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EveryNTimesteps,
    CallbackList,
    BaseCallback,
)
import gymnasium as gym


import hydra
import torch as th

import utils
import logger as _logger
import env as _env
import policies


class Trainer:
    def __init__(
        self,
        model,
        name,
        *,
        total_timesteps,
        log_interval,
        save_interval,
    ):
        self.model = model
        self.name = name

        self.total_timesteps = total_timesteps
        self.log_interval = log_interval
        self.save_interval = save_interval

    def train(self, callbacks=[]):
        # wandb_callback = WandbCallback(
        #     gradient_save_freq=self.save_interval,
        #     model_save_path=f"models/{self.name}",
        #     verbose=2,
        # )

        checkpoint_callback = CheckpointCallback(
            save_freq=self.save_interval,
            verbose=2,
            save_path=f"models/{self.name}/logs",
        )

        self.model.learn(
            total_timesteps=self.total_timesteps,
            progress_bar=True,
            reset_num_timesteps=False,
            # callback=CallbackList([wandb_callback, checkpoint_callback]),
            callback=CallbackList([checkpoint_callback, *callbacks]),
        )


@hydra.main(version_base=None, config_path="exp", config_name="train")
def train(config: DictConfig) -> None:
    exp_cfg = config.exp

    # initialize environment
    env, val_env = getattr(_env.utils, exp_cfg.env_gen_fn)(**exp_cfg.env_gen_fn_params)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    # initialize model
    device = "cuda" if th.cuda.is_available() else "cpu"
    device = th.device(device)
    print(f"""[INFO] Using device: {device}""")

    # get run name
    run_name = utils.get_run_name(exp_cfg)

    model = getattr(policies, exp_cfg.policy)(
        "MultiInputPolicy",
        env,
        **exp_cfg.policy_params,
        device=device,
        tensorboard_log=f"logs/{run_name}",
    )

    # check for checkpoint
    # if "ckpt" in config:
    #     ckpt = model.load(str(config.ckpt))
    #     run_name = ckpt["name"]

    # logger
    logger = _logger.WandbLogger(run_name, config=exp_cfg)

    # initialize Trainer
    trainer = Trainer(
        model=model,
        name=run_name,
        **exp_cfg.trainer_params,
    )

    # _, best_val, scores = _env.utils.get_optimal_path(env.graph)
    # trainer.train(
    #     callbacks=[
    #         policies.ValCallback(
    #             val_env, scores, log_freq=exp_cfg.trainer_params.log_interval
    #         )
    #     ]
    # )

    mult_scores = []
    for _val_env in val_env:
        _, best_val, scores = _env.utils.get_optimal_path(_val_env.graph)
        mult_scores.append(scores)
    trainer.train(
        callbacks=[
            policies.RandomValCallback(
                val_env, mult_scores, log_freq=exp_cfg.trainer_params.log_interval
            )
        ]
    )


if __name__ == "__main__":
    train()
