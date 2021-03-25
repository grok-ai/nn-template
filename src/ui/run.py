from pathlib import Path

import streamlit as st

from src.pl_modules.model import MyModel
from src.ui.ui_utils import select_checkpoint


@st.cache(allow_output_mutation=True)
def get_model(checkpoint_path: Path):
    return MyModel.load_from_checkpoint(checkpoint_path=str(checkpoint_path))


checkpoint_path = select_checkpoint()
model: MyModel = get_model(checkpoint_path=checkpoint_path)
