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

If you already know [cookiecutter](https://github.com/cookiecutter/cookiecutter), just generate your project with:

```bash
cookiecutter https://github.com/grok-ai/nn-template
```

<details>
<summary>Otherwise</summary>
Cookiecutter manages the setup stages and delivers to you a personalized ready to run project. 

Install it with:
<pre><code>pip install cookiecutter
</code></pre>
</details>

More details in the [documentation](https://grok-ai.github.io/nn-template/latest/getting-started/generation/).

## Strengths  
- **Actually works for [research](https://grok-ai.github.io/nn-template/latest/papers/)**!
- Guided setup to customize project bootstrapping;
- Fast prototyping of new ideas, no need to build a new code base from scratch;
- Less boilerplate with no impact on the learning curve (as long as you know the integrated tools);
- Automatize via GitHub actions: testing, stylish documentation deploy, PyPi upload;
- Enforce Python [best practices](https://grok-ai.github.io/nn-template/latest/features/bestpractices/);

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

