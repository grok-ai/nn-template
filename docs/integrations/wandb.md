
# Weights and Biases

Weights & Biases helps you keep track of your machine learning projects. Use tools to log hyperparameters and output metrics from your runs, then visualize and compare results and quickly share findings with your colleagues.

[This](https://wandb.ai/gladia/nn-template?workspace=user-lucmos) is an example of a simple dashboard.

## Quickstart

Login to your `wandb` account, running once `wandb login`.
Configure the logging in `conf/logging/*`.

!!! info

    Read more in the [docs](https://docs.wandb.ai/). Particularly useful the [`log` method](https://docs.wandb.ai/library/log), accessible from inside a PyTorch Lightning module with `self.logger.experiment.log`.
