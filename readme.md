# ml-template

Quick start template for ML experiments with easy experimentation and logging via hydra and wandb.

To run an experiment, run `python train.py +exp=exp0` to run `exp0.yaml` (found in `exp/exp` folder).

To run from a checkpoint, run `python train.py +exp=exp0 +ckpt=*checkpoint_file*`.

**Note: random graphs created by env.utils are metric**

_In get_list, nodes with probabilities of 0 other than start node should be placed at the end._
