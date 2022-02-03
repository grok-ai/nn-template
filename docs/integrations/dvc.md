
# Data Version Control

DVC runs alongside `git` and uses the current commit hash to version control the data.

Initialize the `dvc` repository:

```bash
$ dvc init
```

To start tracking a file or directory, use `dvc add`:

```bash
$ dvc add data/ImageNet
```

DVC stores information about the added file (or a directory) in a special `.dvc` file named `data/ImageNet.dvc`, a small text file with a human-readable format.
This file can be easily versioned like source code with Git, as a placeholder for the original data (which gets listed in `.gitignore`):

```bash
git add data/ImageNet.dvc data/.gitignore
git commit -m "Add raw data"
```

## Making changes

When you make a change to a file or directory, run `dvc add` again to track the latest version:

```bash
$ dvc add data/ImageNet
```

## Switching between versions

The regular workflow is to use `git checkout` first to switch a branch, checkout a commit, or a revision of a `.dvc` file, and then run `dvc checkout` to sync data:

```bash
$ git checkout <...>
$ dvc checkout
```

!!! info

    Read more in the DVC [docs](https://dvc.org/doc/start/data-versioning)!
