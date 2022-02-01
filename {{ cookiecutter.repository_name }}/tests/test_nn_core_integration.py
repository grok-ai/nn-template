from nn_core.common import PROJECT_ROOT


def test_project_root() -> None:
    assert PROJECT_ROOT
    assert (PROJECT_ROOT / "conf").exists()
