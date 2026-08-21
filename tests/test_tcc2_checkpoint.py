import pytest

from tcc_ecg.tcc2_training import load_training_checkpoint, save_training_checkpoint


def test_checkpoint_roundtrip_and_config_guard(tmp_path):
    torch = pytest.importorskip("torch")
    config = {"project": {"seed": 42}, "project_root": str(tmp_path)}
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    path = tmp_path / "checkpoint.pt"
    save_training_checkpoint(path, model, optimizer, epoch=3, best_score=0.75, config=config)
    loaded = load_training_checkpoint(path, model, optimizer, config, torch.device("cpu"))
    assert loaded["epoch"] == 3
    assert loaded["best_score"] == 0.75
    changed = {"project": {"seed": 123}, "project_root": str(tmp_path)}
    with pytest.raises(ValueError):
        load_training_checkpoint(path, model, optimizer, changed, torch.device("cpu"))

