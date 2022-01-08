import logging
import os
from pathlib import Path

import git

from nn_core.common.utils import load_envs

logger = logging.getLogger(__name__)


# Load environment variables
load_envs()

try:
    PROJECT_ROOT = Path(git.Repo(Path.cwd(), search_parent_directories=True).working_dir)
except git.exc.InvalidGitRepositoryError:
    PROJECT_ROOT = Path.cwd()

logger.debug(f"Inferred project root: {PROJECT_ROOT}")
os.environ["PROJECT_ROOT"] = str(PROJECT_ROOT)

Split = str
