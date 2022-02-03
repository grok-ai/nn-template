
# Hydra

Hydra is an open-source Python framework that simplifies the development of research and other complex applications. The key feature is the ability to dynamically create a hierarchical configuration by composition and override it through config files and the command line. The name Hydra comes from its ability to run multiple similar jobs - much like a Hydra with multiple heads.

The basic functionalities are intuitive: it is enough to change the configuration files in `conf/*` accordingly to your preferences. Everything will be logged in `wandb` automatically.

Consider creating new root configurations `conf/myawesomeexp.yaml` instead of always using the default `conf/default.yaml`.


## Sweeps

You can easily perform hyperparameters [sweeps](https://hydra.cc/docs/advanced/override_grammar/extended), which override the configuration defined in `/conf/*`.

The easiest one is the grid-search. It executes the code with every possible combinations of the specified hyperparameters:

```bash
PYTHONPATH=. python src/run.py -m optim.optimizer.lr=0.02,0.002,0.0002 optim.lr_scheduler.T_mult=1,2 optim.optimizer.weight_decay=0,1e-5
```

You can explore aggregate statistics or compare and analyze each run in the W&B dashboard.


!!! info

    We recommend to go through at least the [Basic Tutorial](https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli), and the docs about [Instantiating objects with Hydra](https://hydra.cc/docs/patterns/instantiate_objects/overview).
