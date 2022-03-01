
# Environment Variables

System specific variables (e.g. absolute paths) should not be under version control, otherwise there will be conflicts between different users.

The best way to handle system specific variables is through environment variables.

You can define new environment variables in a `.env` file in the project root. A copy of this file (e.g. `.env.template`) can be under version control to ease new project configurations.

To define a new variable write inside `.env`:

```bash
export MY_VAR=/home/user/my_system_path
```


You can dynamically resolve the variable name from everywhere

=== "python"

    In Python code use:

    ```python
    get_env("MY_VAR")
    ```

=== "yaml"

    In the Hydra `yaml` configurations:

    ```yaml
    ${oc.env:MY_VAR}
    ```

=== "posix"

    In posix shells:

    ```bash
    . .env
    echo $MY_VAR
    ```
