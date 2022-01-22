import shutil


def initialize_env_variables(
    env_file: str = ".env", env_file_template: str = ".env.template"
) -> None:
    """Initialize the .env file"""
    shutil.copy(src=env_file_template, dst=env_file)


initialize_env_variables()
