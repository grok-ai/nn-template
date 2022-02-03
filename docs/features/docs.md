# Documentation

`MkDocs` and `Material for MkDocs` is already configured in the generated project.

In order to create your docs it is enough to:

1. Modify the `nav` index in the `mkdocs.yaml`, which describes how to organize the pages.
   An example of the `nav` is the following:

    ```yaml
    nav:
      - Home: index.md
      - Getting started:
        - Generating your project: getting-started/generation.md
        - Strucure: getting-started/structure.md
    ```

2. Create all the files referenced in the `nav` relative to the `docs/` folder.

    ```bash
    ❯ tree docs
    docs
    ├── getting-started
    │   ├── generation.md
    │   └── structure.md
    └── index.md
    ```

3. To preview your documentation it is enough to run `mkdocs serve`. To manually deploy the documentation
    see [`mike`](https://github.com/jimporter/mike), or see the integrated GitHub Action to [publish the docs on release](/features/cicd/#publish-docs).
