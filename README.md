# NN Template

<p align="center">
    <a href="https://github.com/grok-ai/nn-template/actions/workflows/test_suite.yml"><img alt="CI" src=https://img.shields.io/github/workflow/status/grok-ai/nn-template/Test%20Suite/main?label=main%20checks></a>
    <a href="https://github.com/grok-ai/nn-template/actions/workflows/test_suite.yml"><img alt="CI" src=https://img.shields.io/github/workflow/status/grok-ai/nn-template/Test%20Suite/develop?label=develop%20checks></a>
    <a href="https://grok-ai.github.io/nn-template"><img alt="Docs" src=https://img.shields.io/github/deployments/grok-ai/nn-template/github-pages?label=docs></a>
    <a href="https://pypi.org/project/nn-template-core/"><img alt="Release" src="https://img.shields.io/pypi/v/nn-template-core?label=nn-core"></a>
    <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

[comment]: <> (<p align="center">)

[comment]: <> (    <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch-red?logo=pytorch&labelColor=gray"></a>)

[comment]: <> (    <a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/code-Lightning-blueviolet"></a>)

[comment]: <> (    <a href="https://hydra.cc/"><img alt="Conf: hydra" src="https://img.shields.io/badge/conf-hydra-blue"></a>)

[comment]: <> (    <a href="https://wandb.ai/site"><img alt="Logging: wandb" src="https://img.shields.io/badge/logging-wandb-yellow"></a>)

[comment]: <> (    <a href="https://dvc.org/"><img alt="Conf: hydra" src="https://img.shields.io/badge/data-dvc-9cf"></a>)

[comment]: <> (    <a href="https://streamlit.io/"><img alt="UI: streamlit" src="https://img.shields.io/badge/ui-streamlit-orange"></a>)

[comment]: <> (</p>)

<p align="center">
    <i>
        nn-template is opinionated so you don't have to be
    </i>
</p>


Generic template to bootstrap your [PyTorch](https://pytorch.org/get-started/locally/) project,
read more in the [documentation](https://grok-ai.github.io/nn-template).

## Get started

Generate your project with cookiecutter:

```bash
cookiecutter https://github.com/grok-ai/nn-template
```

> This is a *parametrized* template that uses [cookiecutter](https://github.com/cookiecutter/cookiecutter).
> Install cookiecutter with:
>
> ```pip install cookiecutter```


## Integrations

Avoid writing boilerplate code to integrate:

- [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning), lightweight PyTorch wrapper for high-performance AI research.
- [Hydra](https://github.com/facebookresearch/hydra), a framework for elegantly configuring complex applications.
- [Weights and Biases](https://wandb.ai/home), organize and analyze machine learning experiments. *(educational account available)*
- [Streamlit](https://streamlit.io/), turns data scripts into shareable web apps in minutes.
- [MkDocs](https://www.mkdocs.org/) and [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/), a fast, simple and downright gorgeous static site generator.
- [DVC](https://dvc.org/doc/start/data-versioning), track large files, directories, or ML models. Think "Git for data".
- [GitHub Actions](https://github.com/features/actions), to run the tests, publish the documentation and to PyPI automatically.
- Python best practices for developing and publishing research projects.

## Structure

The generated projects will contain the following files:

```bash
.
├── conf
│   ├── default.yaml
│   ├── hydra
│   │   └── default.yaml
│   ├── nn
│   │   └── default.yaml
│   └── train
│       └── default.yaml
├── data
│   └── .gitignore
├── docs
│   ├── index.md
│   └── overrides
│       └── main.html
├── .editorconfig
├── .env
├── .env.template
├── env.yaml
├── .flake8
├── .github
│   └── workflows
│       ├── publish.yml
│       └── test_suite.yml
├── .gitignore
├── LICENSE
├── mkdocs.yml
├── .pre-commit-config.yaml
├── pyproject.toml
├── README.md
├── setup.cfg
├── setup.py
├── src
│   └── awesome_project
│       ├── data
│       │   ├── datamodule.py
│       │   ├── dataset.py
│       │   └── __init__.py
│       ├── __init__.py
│       ├── modules
│       │   ├── __init__.py
│       │   └── module.py
│       ├── pl_modules
│       │   ├── __init__.py
│       │   └── pl_module.py
│       ├── run.py
│       └── ui
│           ├── __init__.py
│           └── run.py
└── tests
    ├── conftest.py
    ├── __init__.py
    ├── test_checkpoint.py
    ├── test_configuration.py
    ├── test_nn_core_integration.py
    ├── test_resume.py
    ├── test_storage.py
    └── test_training.py
```
