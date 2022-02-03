# {{ cookiecutter.project_name }}

<p align="center">
    <a href="https://github.com/{{ cookiecutter.github_user }}/{{ cookiecutter.repository_name }}/actions/workflows/test_suite.yml"><img alt="CI" src=https://img.shields.io/github/workflow/status/{{ cookiecutter.github_user }}/{{ cookiecutter.repository_name }}/Test%20Suite/main?label=main%20checks></a>
    <a href="https://{{ cookiecutter.github_user }}.github.io/{{ cookiecutter.repository_name }}"><img alt="Docs" src=https://img.shields.io/github/deployments/{{ cookiecutter.github_user }}/{{ cookiecutter.repository_name }}/github-pages?label=docs></a>
    <a href="https://github.com/grok-ai/nn-template"><img alt="NN Template" src="https://shields.io/badge/nn--template-{{ cookiecutter.__version }}-emerald?style=flat&labelColor=gray"></a>
    <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-{{ cookiecutter.python_version }}-blue.svg"></a>
    <a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

{{ cookiecutter.project_description }}


## Installation

```bash
pip install git+ssh://git@github.com/{{ cookiecutter.github_user }}/{{ cookiecutter.repository_name }}.git
```


## Quickstart

[comment]: <> (> Fill me!)


## Development installation

Setup the development environment:

```bash
git clone git+ssh://git@grok-ai/{{ cookiecutter.github_user }}/{{ cookiecutter.repository_name }}.git
conda env create -f env.yaml
conda activate {{ cookiecutter.conda_env_name }}
pre-commit install
```

Run the tests:

```bash
pre-commit run --all-files
pytest -v
```


### Update the dependencies

Re-install the project in edit mode:

```bash
pip install -e .[dev]
```
