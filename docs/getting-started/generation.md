# Initial Setup

## Cookiecutter

`nn-template` is, by definition, a template to generate projects. It's a robust **starting point**
for your projects, something that lets you skip the initial boilerplate in configuring the environment, tests and such.
Since it is a blueprint to build upon, it has no utility in being installed via pip or similar tools.

Instead, we rely on [cookiecutter](https://cookiecutter.readthedocs.io) to manage the setup stages and deliver to you a
ready-to-run project. It is a general-purpose tool that enables users to add their water of choice (variable
configurations) to their particular Cup-a-Soup (the template to be setup).

!!! hint "Installing cookiecutter"

    `cookiecutter` can be installed via pip in any Python-enabled environment (it won't be the same used by the project once
    instantiated). Our advice is to install `cookiecutter` as a system utility via [pipx](https://github.com/pypa/pipx):

    ```shell
    pipx install cookiecutter
    ```

Then, we need to tell cookiecutter which template to work on:

```shell
cookiecutter https://github.com/grok-ai/nn-template.git
```

It will clone the nn-template repository in the background, call its interactive setup, and build your project's folder
according to the given parametrization.

The parametrized setup will take care of:

- Set up the development of a Python package
- Initializing a clean Git repository and add the GitHub remote of choice
- Create a **new Conda environment** to execute your code in

This extra step via cookiecutter is done to avoid a lot of manual parametrization, unavoidable when cloning a template
repository from scratch. **Trust us, it is totally worth the bother!**

## Building Blocks
The generated project already contains a minimal working example. You are **free to modify anything** you want except for
a few essential and high-level things that keep everything working. **(again, this is not a framework!)**.
In particular mantain:

- Any `LightningLogger` you may want to use wrapped in a `NNLogger`
- The `NNTemplateCore` Lightning callback

!!! hint nn-template main components

    The template bootstraps the project with most of the needed boilerplate.
    The remaining components to implement for your project are the following:

    1. Implement data pipeline
        1. Dataset
        2. Pytorch Lightning DataModule
    2. Implement neural modules
        1. Model
        2. Pytorch Lightning Module

## FAQs

??? question "What is The Answer to the Ultimate Question of Life, the Universe, and Everything?"

    42

??? question "Why are the logs badly formatted in PyCharm?"

    This is due to the fact that we are using [Rich](https://rich.readthedocs.io/en/stable/introduction.html) to handle
    the logging, and Rich is not compatible with customized terminals. As its documentation says:

    "*PyCharm users will need to enable “emulate terminal” in output console option in run/debug configuration to see styled output.*"

??? question "Why are file paths not interactive in the terminal's output?"

    [We would like to know, too.](https://youtrack.jetbrains.com/issue/PY-46305)

??? question "How can I exclude specific file paths from pre-commit checks (e.g. pydocstyle)?"

    While we encourage everyone to keep best-practices and standards enforced via the pre-commit utility, we also take
    into account situations where you just copy/paste code from the Internet and fixing it would be tedious.
    In those cases, the file `.pre-commit-config.yaml` has you covered. Each hook can receive an additional property,
    namely `exclude` where you can specify single files or patterns to be excluded when running that hook.

    For example, if you want to exclude a file named `ugly_but_working_code.py` from an annoying hook `annoying_hook` (most likely `pydocstyle`):
    ```yaml
      - repo: https://github.com/slow_coding/annoying_hook.git
        hooks:
        -   id: annoying_hook
            exclude: ugly_but_working_code.py
    ```

## Future Features

- [ ] Optuna support
- [ ] Support different loggers other than WandB
