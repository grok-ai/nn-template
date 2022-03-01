# Tags

Each run should be `tagged` in order to easily filter them from the logged dashboard.
Unfortunately, it is easy to forget to tag correctly each run.

We ask interactively for a list of comma separated tags, if those are not already defined in the configuration:
```
WARNING  No tags provided, asking for tags...
Enter a list of comma separated tags (develop):
```

!!! info

    If the current experiment is a sweep comprised of multiple runs and there are not any tags defined,
    an error is raised instead:
    ```
    ERROR    You need to specify 'core.tags' in a multi-run setting!
    ```
