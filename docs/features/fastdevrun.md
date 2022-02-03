# Fast Dev Run

The template expands the Lightning `fast_dev_run` mode to be more debugging friendly.

It will also:

- Disable multiple workers in the dataloaders
- Use the CPU and not the GPU

!!! info

    It is possible to modify this behaviour by simply modifying the `run.py` file.
