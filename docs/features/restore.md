# Restore

The template offers a way to restore a previous run from the configuration.
The relevant configuration block is in `conf/train/default.yml`:

```yaml
restore:
  ckpt_or_run_path: null
  mode: null # null, continue, hotstart
```

## ckpt_or_run_path

The `ckpt_or_run_path` can be a path towards a Lightning Checkpoint or the run identifies.
In case of W&B it is called `run_path` and are in the form of `entity/project/run_id`.

!!! warning

    If `ckpt_or_run_path` points to a checkpoint, that checkpoint must have been saved with
    this template, because additional information are attached to the checkpoint to guarantee
    a correct restore. These include the `run_path` itself and the whole configuration used.

## mode

We support three different modes for restoring an experiment:

=== "continue"

    ```yaml
    restore:
        mode: continue
    ```
    In this `mode` the training continues from the checkpoint **and** the logging continues
    in the previous run. No new run is created on the logger dashboard.

=== "hotstart"

    ```yaml
    restore:
        mode: hotstart
    ```
    In this `mode` the training continues from the checkpoint **but** the logging does not.
    A new run is created on the logger dashboard.

=== "null"

    ```yaml
    restore:
        mode: null
    ```
    In this `mode` no restore happens, and `ckpt_or_run_path` is ignored.
