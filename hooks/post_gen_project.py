import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from distutils.util import strtobool
from typing import Dict, List, Optional


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


@dataclass
class Dependency:
    expected: bool
    id: str


@dataclass
class Query:
    id: str
    interactive: bool
    default: bool
    prompt: str
    command: str
    autorun: bool
    dependencies: List[Dependency] = field(default_factory=list)


SETUP_COMMANDS: List[Query] = [
    Query(
        id="git_init",
        interactive=True,
        default=True,
        prompt="Initializing git repository...",
        command="git init\n"
        "git add --all\n"
        'git commit -m "Initialize project from nn-template={{ cookiecutter.__version }}"',
        autorun=True,
    ),
    Query(
        id="git_remote",
        interactive=True,
        default=True,
        prompt="Adding an existing git remote...",
        command="git remote add origin git@github.com:{{ cookiecutter.github_user }}/{{ cookiecutter.repository_name }}.git",
        autorun=True,
        dependencies=[
            Dependency(id="git_init", expected=True),
        ],
    ),
    Query(
        id="git_push_main",
        interactive=True,
        default=True,
        prompt="Pushing default branch to existing remote...",
        command="git push -u origin HEAD",
        autorun=True,
        dependencies=[
            Dependency(id="git_remote", expected=True),
        ],
    ),
    Query(
        id="conda_env",
        interactive=True,
        default=True,
        prompt="Creating conda environment...",
        command="conda env create -f env.yaml",
        autorun=True,
    ),
    Query(
        id="precommit_install",
        interactive=True,
        default=True,
        prompt="Installing pre-commits...",
        command="conda run -n {{ cookiecutter.conda_env_name }} pre-commit install",
        autorun=True,
        dependencies=[
            Dependency(id="git_init", expected=True),
            Dependency(id="conda_env", expected=True),
        ],
    ),
    Query(
        id="mike_init",
        interactive=True,
        default=True,
        prompt="Initializing gh-pages branch for GitHub Pages...",
        command="conda run -n {{ cookiecutter.conda_env_name }} mike deploy --update-aliases 0.0 latest\n"
        "conda run -n {{ cookiecutter.conda_env_name }} mike set-default latest",
        autorun=True,
        dependencies=[
            Dependency(id="conda_env", expected=True),
            Dependency(id="git_init", expected=True),
        ],
    ),
    Query(
        id="mike_push",
        interactive=True,
        default=True,
        prompt="Pushing 'gh-pages' branch to existing remote...",
        command="git push origin gh-pages",
        autorun=True,
        dependencies=[
            Dependency(id="mike_init", expected=True),
            Dependency(id="git_remote", expected=True),
        ],
    ),
    Query(
        id="conda_activate",
        interactive=False,
        default=True,
        prompt="Activate your conda environment with:",
        command="cd {{ cookiecutter.repository_name }}\n"
        "conda activate {{ cookiecutter.conda_env_name }}\n"
        "pytest -v",
        autorun=False,
        dependencies=[
            Dependency(id="conda_env", expected=True),
        ],
    ),
]


def should_execute_query(query: Query, answers: Dict[str, bool]) -> bool:
    if not query.dependencies:
        return True
    return all(
        dependency.expected == answers.get(dependency.id, False)
        for dependency in query.dependencies
    )


def setup(setup_commands) -> None:
    answers: Dict[str, bool] = {}

    for query in setup_commands:
        assert query.id not in answers

        if should_execute_query(query=query, answers=answers):
            if query.interactive:
                answers[query.id] = bool_query(
                    question=f"\n"
                    f"{query.prompt}\n"
                    f"\n"
                    f'{textwrap.indent(query.command, prefix="    ")}\n'
                    f"\n"
                    f"Execute?",
                    default=query.default,
                )
            else:
                print(
                    f"\n"
                    f"{query.prompt}\n"
                    f"\n"
                    f'{textwrap.indent(query.command, prefix="    ")}\n'
                )
                answers[query.id] = True

            if answers[query.id] and (query.interactive or query.autorun):
                try:
                    subprocess.run(
                        query.command,
                        check=True,
                        text=True,
                        shell=True,
                    )
                except subprocess.CalledProcessError:
                    answers[query.id] = False
            print()


initialize_env_variables()
setup(setup_commands=SETUP_COMMANDS)

print(
    "\nYou are all set!\n\n"
    "Remember that if you use PyCharm, you must:\n"
    '    - Mark the "src" directory as "Sources Root".\n'
    '    - Enable "Emulate terminal in output console" in the run configuration.\n'
)
print(
    "Remember to:\n"
    "    - Ensure the GitHub Actions in the repository are enabled.\n"
    "    - Ensure the Github Pages are enabled to auto-publish the docs on each release."
)

print("Have fun! :]")
