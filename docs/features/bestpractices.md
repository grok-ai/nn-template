# Tooling

The template configures are the tooling necessary for a modern python project.

These include:

- [**EditorConfig**](https://editorconfig.org/)  maintain consistent coding styles for multiple developers.
- [**Black**](https://black.readthedocs.io/en/stable/index.html) the uncompromising code formatter.
- [**isort**](https://github.com/PyCQA/isort) sort imports alphabetically, and automatically separated into sections and by type.
- [**flake8**](https://flake8.pycqa.org/en/latest/) check coding style (PEP8), programming errors and cyclomatic complexity.
- [**pydocstyle**](http://www.pydocstyle.org/en/stable/) static analysis tool for checking compliance with Python docstring conventions.
- [**MyPy**](http://mypy-lang.org/) static type checker for Python.
- [**Coverage**](https://coverage.readthedocs.io/en/6.2/) measure code coverage of Python programs.
- [**bandit**](https://github.com/PyCQA/bandit) security linter from PyCQA.
- [**pre-commit**](https://pre-commit.com/) framework for managing and maintaining pre-commit hooks.

## Pre commits

The pre-commits configuration is defined in `.pre-commit-config.yaml`,
and includes the most important checks and auto-fix to perform.

If one of the pre-commits fails, **the commit is aborted** avoiding distraction errors.

!!! info

    The pre-commits are also run in the CI/CD as part of the Test Suite. This helps
    guaranteeing the all the contributors are respecting the code conventions in the
    pull requests.
