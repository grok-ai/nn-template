
# Structure

```bash
.
├── .cache
├── conf                # hydra compositional config
│   ├── nn
│   ├── default.yaml    # current experiment configuration
│   ├── hydra
│   └── train
├── data                # datasets
├── .env                # system-specific env variables, e.g. PROJECT_ROOT
├── requirements.txt    # basic requirements
├── src
│   ├── common          # common modules and utilities
│   ├── data         # PyTorch Lightning datamodules and datasets
│   ├── modules      # PyTorch Lightning modules
│   ├── run.py          # entry point to run current conf
│   └── ui              # interactive streamlit apps
└── wandb               # local experiments (auto-generated)
```
