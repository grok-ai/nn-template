
# Structure

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