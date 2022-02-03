# Storage

The checkpoints and other data produces by the experiment is stored in
a logger agnostic folder defined in the configuration `core.storage_dir`

This is the organization of the `storage_dir`:

```bash
storage
└── <project_name>
    └──  <run_id>
         ├── checkpoints
         │    └── <checkpoint_name>.ckpt
         └── config.yaml
```

In the configuration it is possible to specify whether the run files
stored inside the `storage_dir` should be uploaded to the cloud:

```yaml
logging:
  upload:
    run_files: true
```
