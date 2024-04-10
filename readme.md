# ml-template 

Quick start template for ML experiments with easy experimentation and logging via hydra and wandb.

To run an experiment, run `python train.py +exp=exp0` to run `exp0.yaml` (found in `exp/exp` folder).

To run from a checkpoint, run `python train.py +exp=exp0 +ckpt=*checkpoint_file*`.
