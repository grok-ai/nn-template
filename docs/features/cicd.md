# CI/CD

The generated project contains two GiHub Actions workflow to run the Test Suite and to publish you project.

!!! note
    You need to enable the GitHub Actions from the settings in your repository.

!!! important
    All the workflow already implement the logic needed to **cache** the conda and pip environment
    between workflow runs.

!!! warning
    The annotated tags in the git repository to manage releases should follow the [semantic versioning](https://semver.org/l)
    conventions: `<major>.<minor>.<patch>`



## Test Suite

The Test Suite runs automatically for each commit in a Pull Request.
It is successful if:

- The pre-commits do not raise any errors
- All the tests pass

After that, the PR are marked with ✔️ or ❌ depending on the test suite results.

## Publish docs

The first time you should use `mike` to: create the `gh-pages` branch and
specify the default docs version.

```bash
mike deploy 0.1 latest --push
mike set-default latest
```

!!! info

    Remember to enable the GitHub Pages from the repository settings.


After that, the docs are built and automatically published `on release` on GitHub Pages.
This means that every time you publish a new release in your project an associated version of the documentation is published.

!!! important

    The documentation version utilizes only the `<major>.<minor>` version of the release tag, discarding the patch version.

## Publish PyPi

To publish your package on PyPi it is enough to configure
the PyPi token in the GitHub repository `secrets` and de-comment the following in the
`publish.yaml` workflow:

```yaml
      - name: Build SDist and wheel
        run: pipx run build

      - name: Check metadata
        run: pipx run twine check dist/*

      - name: Publish distribution 📦 to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          user: __token__
          password: ${{ secrets.PYPI_API_TOKEN }}
```

In this way, on each GitHub release the package gets published on PyPi and the associated documentation
is published on GitHub Pages.
