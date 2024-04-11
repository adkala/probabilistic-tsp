from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from scipy.stats import percentileofscore

import env


class ValCallback(BaseCallback):
    """
    Custom callback for plotting validation values to tensorboard.
    """

    def __init__(self, val_env, scores, log_freq=1000, verbose: int = 0):
        super().__init__(verbose)
        self.val_env = val_env
        self.scores = scores
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq == 0:
            cur_path = env.utils.get_policy_path(self.model, self.val_env)
            exp_dist = env.utils.get_expected_dist(self.val_env.graph, cur_path)
            self.logger.record(
                "val/exp_dist",
                exp_dist,
                self.num_timesteps,
            )
            self.logger.record(
                "val/exp_dist_perc",
                1 - percentileofscore(self.scores, exp_dist) / 100,
                self.num_timesteps,
            )
        return True


class RandomValCallback(BaseCallback):
    def __init__(self, val_envs, mult_scores, log_freq=1000, verbose: int = 0):
        super().__init__(verbose)
        self.val_envs = val_envs
        self.mult_scores = mult_scores
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq == 0:
            exp_dist_cum = 0
            exp_dist_perc_cum = 0
            for val_env, scores in zip(self.val_envs, self.mult_scores):
                cur_path = env.utils.get_policy_path(self.model, val_env)
                exp_dist = env.utils.get_expected_dist(val_env.graph, cur_path)
                exp_dist_cum += exp_dist
                exp_dist_perc_cum += 1 - percentileofscore(scores, exp_dist) / 100
            self.logger.record(
                "val/exp_dist",
                exp_dist_cum / len(self.val_envs),
                self.num_timesteps,
            )
            self.logger.record(
                "val/exp_dist_perc",
                exp_dist_perc_cum / len(self.val_envs),
                self.num_timesteps,
            )
        return True
