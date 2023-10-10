---
hide:
  - navigation
  - toc
---

# NN Template

<p align="center">
    <a href="https://github.com/grok-ai/nn-template/actions/workflows/test_suite.yml"><img alt="CI" src=https://github.com/grok-ai/nn-template/actions/workflows/test_suite.yml/badge.svg?branch=main></a>
    <a href="https://github.com/grok-ai/nn-template/actions/workflows/test_suite.yml"><img alt="CI" src=https://github.com/grok-ai/nn-template/actions/workflows/test_suite.yml/badge.svg?branch=develop></a>
    <a href="https://pypi.org/project/nn-template-core/"><img alt="Release" src="https://img.shields.io/pypi/v/nn-template-core?label=nn-core"></a>
    <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

<p align="center">
    <i>
        "We demand rigidly defined areas of doubt and uncertainty."
    </i>
</p>

---

```bash
cookiecutter https://github.com/grok-ai/nn-template
```

---

[![asciicast](https://asciinema.org/a/475623.svg)](https://asciinema.org/a/475623)

---

Generic cookiecutter template to bootstrap [PyTorch](https://pytorch.org/get-started/locally/) projects
and to avoid writing boilerplate code to integrate:

- [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning), lightweight PyTorch wrapper for high-performance AI research.
- [Hydra](https://github.com/facebookresearch/hydra), a framework for elegantly configuring complex applications.
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/index),a library for easily accessing and sharing datasets.
- [Weights and Biases](https://wandb.ai/home), organize and analyze machine learning experiments. *(educational account available)*
- [Streamlit](https://streamlit.io/), turns data scripts into shareable web apps in minutes.
- [MkDocs](https://www.mkdocs.org/) and [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/), a fast, simple and downright gorgeous static site generator.
- [DVC](https://dvc.org/doc/start/data-versioning), track large files, directories, or ML models. Think "Git for data".
- [GitHub Actions](https://github.com/features/actions), to run the tests, publish the documentation and to PyPI automatically.
- Python best practices for developing and publishing research projects.


!!! help "cookiecutter"

    This is a *parametrized* template that uses [cookiecutter](https://github.com/cookiecutter/cookiecutter).
    Install cookiecutter with:

    ```pip install cookiecutter```
