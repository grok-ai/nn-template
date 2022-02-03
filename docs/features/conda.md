# Python Environment

The generated project is a Python Package, whose dependencies are defined in the `setup.cfg`
The development setup comprises a `conda` environment which installs the package itself in edit mode.

## Dependencies

All the project dependencies should be defined in the `setup.cfg` as `pip` dependencies.
In rare cases, it is useful to specify conda dependencies --- they will not be resolved when installing the package
from PyPi.

This division is useful when installing particular or optimized packages such a `PyTorch` and PyTorch Geometric.

!!! hint

    It is possible to manage the Python version to use in the conda `env.yaml`.

!!! info

    This organization allows for `conda` and `pip` dependencies to co-exhist, which in practice happens a lot in
    research projects.

## Update

In order to update the `pip` dependencies after changing the `setup.cfg` it is enough to run:

```bash
pip install -e '.[dev]'
```
