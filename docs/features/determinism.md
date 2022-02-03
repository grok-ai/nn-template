# Determinism

The template always logs the seed utilized in order to guarantee reproducibility.

The user specifies a `seed_index` value in the configuration `train/default.yaml`:

```yaml
seed_index: 0
deterministic: False
```

This value indexes an array of deterministic but randomly generated seeds, e.g.:

```bash
Setting seed 1273642419 from seeds[1]
```


!!! hint

    This setup allows to easily run the same experiment with different seeds in a reproducible way.
    It is enough to run a Hydra multi-run over the `seed_index`.

    The following would run the same experiment with five different seeds, which can be analyzed
    in the logger dashboard:

    ```bash
    python src/project/run.py -m train.seed_index=0,1,2,3,4
    ```

!!! info

    The deterministic option `deterministic: False` controls the use of deterministic algorithms
    in PyTorch, it is forwarded to the Lightning Trainer.
