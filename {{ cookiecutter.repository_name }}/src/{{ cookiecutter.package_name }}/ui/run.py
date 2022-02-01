from pathlib import Path

import streamlit as st
import wandb

from nn_core.serialization import NNCheckpointIO
from nn_core.ui import select_checkpoint

from {{ cookiecutter.package_name }}.pl_modules.pl_module import MyLightningModule


@st.cache(allow_output_mutation=True)
def get_model(checkpoint_path: Path):
    return NNCheckpointIO.load(path=checkpoint_path)


if wandb.api.api_key is None:
    st.error("You are not logged in on `Weights and Biases`: https://docs.wandb.ai/ref/cli/wandb-login")
    st.stop()

st.sidebar.subheader(f"Logged in W&B as: {wandb.api.viewer()['entity']}")

checkpoint_path = select_checkpoint()
model: MyLightningModule = get_model(checkpoint_path=checkpoint_path)
model
