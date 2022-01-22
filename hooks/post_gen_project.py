import shutil
import subprocess
import sys
import textwrap
from distutils.util import strtobool
from typing import List, Optional, Tuple


def initialize_env_variables(
    env_file: str = ".env", env_file_template: str = ".env.template"
) -> None:
    """Initialize the .env file"""
    shutil.copy(src=env_file_template, dst=env_file)


def bool_query(question: str, default: Optional[bool] = None) -> bool:
    """Ask a yes/no question via input() and return their boolean answer.

    Args:
        question: is a string that is presented to the user.
        default: is the presumed answer if the user just hits <Enter>.

    Returns:
        the boolean representation of the user answer, or the default value if present.
    """
    if default is None:
        prompt = " [y/n] "
    elif default:
        prompt = " [Y/n] "
    else:
        prompt = " [y/N] "

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()

        if default is not None and not choice:
            return default

        try:
            return strtobool(choice)
        except ValueError:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


def interactive_setup() -> None:
    setup_question2commands: List[Tuple[str, str]] = [
        (
            "Initializing git repository...",
            "git init\n"
            "git add --all\n"
            'git commit -m "Initialize project from nn-template={{ cookiecutter.__version }}"',
        ),
        (
            "Adding git remote...",
            "git remote add origin git@github.com:{{ cookiecutter.github_user }}/{{ cookiecutter.repository_name }}.git",
        ),
        (
            "Creating conda environment...",
            "conda env create -f env.yaml\n"
            "conda run -n {{ cookiecutter.conda_env_name }} pre-commit install",
        ),
    ]

    for question, command in setup_question2commands:
        if bool_query(
            question=f"\n"
            f"{question}\n"
            f"\n"
            f'{textwrap.indent(command, prefix="    ")}\n'
            f"\n"
            f"Execute?",
            default=True,
        ):
            subprocess.run(
                command,
                check=False,
                text=True,
                shell=True,
            )
        print()

    final_command = (
        "cd {{ cookiecutter.repository_name }}\n"
        "conda activate {{ cookiecutter.conda_env_name }}\n"
    )
    print(
        f"\nActivate your conda environment with:\n"
        f"{textwrap.indent(final_command, prefix='    ')}\n"
    )


initialize_env_variables()
interactive_setup()

print("Have fun! :)")
