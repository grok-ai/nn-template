import datetime
import operator
from pathlib import Path
from typing import List

import hydra
import omegaconf
import streamlit as st
import wandb
from hydra.core.global_hydra import GlobalHydra
from hydra.experimental import compose
from stqdm import stqdm

from src.common.utils import PROJECT_ROOT, load_envs

load_envs()

WANDB_DIR: Path = PROJECT_ROOT / "wandb"
WANDB_DIR.mkdir(exist_ok=True, parents=True)

st_run_sel = st.sidebar


def local_checkpoint_selection(run_dir: Path, st_key: str) -> Path:
    checkpoint_paths: List[Path] = list(run_dir.rglob("checkpoints/*"))
    if len(checkpoint_paths) == 0:
        st.error(
            f"There's no checkpoint under {run_dir}! Are you sure the restore was successful?"
        )
        st.stop()
    checkpoint_path: Path = st_run_sel.selectbox(
        label="Select a checkpoint",
        index=len(checkpoint_paths) - 1,
        options=checkpoint_paths,
        format_func=operator.attrgetter("name"),
        key=f"checkpoint_select_{st_key}",
    )

    return checkpoint_path


def get_run_dir(entity: str, project: str, run_id: str) -> Path:
    """
    :param run_path: "flegyas/nn-template/3hztfivf"
    :return:
    """

    api = wandb.Api()
    run = api.run(path=f"{entity}/{project}/{run_id}")
    created_at: datetime = datetime.datetime.strptime(
        run.created_at, "%Y-%m-%dT%H:%M:%S"
    )
    st.sidebar.markdown(body=f"[`Open on WandB`]({run.url})")

    timestamp: str = created_at.strftime("%Y%m%d_%H%M%S")

    matching_runs: List[Path] = [
        item
        for item in WANDB_DIR.iterdir()
        if item.is_dir() and item.name.endswith(run_id)
    ]

    if len(matching_runs) > 1:
        st.error(
            f"More than one run matching unique id {run_id}! Are you sure about that?"
        )
        st.stop()

    if len(matching_runs) == 1:
        return matching_runs[0]

    only_checkpoint: bool = st_run_sel.checkbox(
        label="Download only the checkpoint?", value=True
    )
    if st_run_sel.button(label="Download"):
        run_dir: Path = WANDB_DIR / f"restored-{timestamp}-{run.id}" / "files"
        files = [
            file
            for file in run.files()
            if "checkpoint" in file.name or not only_checkpoint
        ]
        if len(files) == 0:
            st.error(
                f"There is no file to download from this run! Check on WandB: {run.url}"
            )
        for file in stqdm(files, desc="Downloading files..."):
            file.download(root=run_dir)
        return run_dir
    else:
        st.stop()


def select_run_path(st_key: str, default_run_path: str):
    run_path: str = st_run_sel.text_input(
        label="Run path (entity/project/id):",
        value=default_run_path,
        key=f"run_path_select_{st_key}",
    )
    if not run_path:
        st.stop()
    tokens: List[str] = run_path.split("/")
    if len(tokens) != 3:
        st.error(
            f"This run path {run_path} doesn't look like a WandB run path! Are you sure about that?"
        )
        st.stop()

    return tokens


def select_checkpoint(st_key: str = "MyAwesomeModel", default_run_path: str = ""):
    entity, project, run_id = select_run_path(
        st_key=st_key, default_run_path=default_run_path
    )

    run_dir: Path = get_run_dir(entity=entity, project=project, run_id=run_id)

    return local_checkpoint_selection(run_dir, st_key=st_key)


def get_hydra_cfg(config_name: str = "default") -> omegaconf.DictConfig:
    """
    Instantiate and return the hydra config -- streamlit and jupyter compatible

    Args:
        config_name: .yaml configuration name, without the extension

    Returns:
        The desired omegaconf.DictConfig
    """
    GlobalHydra.instance().clear()
    hydra.experimental.initialize_config_dir(config_dir=str(PROJECT_ROOT / "conf"))
    return compose(config_name=config_name)
