# Restore

The template offers a way to restore a previous run from the configuration.
The relevant configuration block is in `conf/train/default.yml`:

```yaml
restore:
  ckpt_or_run_path: null
  mode: null # null, finetune, hotstart, continue
```

## ckpt_or_run_path

The `ckpt_or_run_path` can be a path towards a Lightning Checkpoint or the run identifiers w.r.t. the logger.
In case of W&B as a logger, they are called `run_path` and are in the form of `entity/project/run_id`.

!!! warning

    If `ckpt_or_run_path` points to a checkpoint, that checkpoint must have been saved with
    this template, because additional information are attached to the checkpoint to guarantee
    a correct restore. These include the `run_path` itself and the whole configuration used.

## mode

We support 4 different modes for restoring an experiment:

=== "null"

    ```yaml
    restore:
        mode: null
    ```
    In this `mode` no restore happens, and `ckpt_or_run_path` is ignored.


    !!! example "Use Case"

        This is the default option and allows the user to train the model from
        scratch logging into a new run.


=== "finetune"

    ```yaml
    restore:
        mode: finetune
    ```
    In this `mode` only the model weights are restored, both the `Trainer` state and the logger run
    are *not restored*.


    !!! example "Use Case"

        As the name suggest, one of the most common use case is when fine
        tuning a trained model logging into a new run with a novel training
        regimen.

=== "hotstart"

    ```yaml
    restore:
        mode: hotstart
    ```
    In this `mode` the training continues from the checkpoint restoring the `Trainer` state **but** the logging does not.
    A new run is created on the logger dashboard.


    !!! example "Use Case"

        Perform different tests in separate logging runs branching from the same trained
        model.


=== "continue"

    ```yaml
    restore:
        mode: continue
    ```
    In this `mode` the training continues from the checkpoint **and** the logging continues
    in the previous run. No new run is created on the logger dashboard.


    !!! example "Use Case"

        The training execution was interrupted and the user wants to continue it.


!!! tldr "Restore summary"

    |               | null | finetune           | hotstart           | continue           |
    |---------------|------|--------------------|--------------------|--------------------|
    | **Model weights** | :x:  | :white_check_mark: | :white_check_mark: | :white_check_mark: |
    | **Trainer state** | :x:  | :x:                | :white_check_mark: | :white_check_mark: |
    | **Logging run**   | :x:  | :x:                | :x:                | :white_check_mark: |
