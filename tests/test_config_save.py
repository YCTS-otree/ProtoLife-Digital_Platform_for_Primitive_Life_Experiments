from pathlib import Path

import yaml

from protolife.config_loader import load_default_config, save_config_with_comments


def test_saved_model_config_keeps_default_layout_and_comments(tmp_path: Path):
    config = load_default_config()
    config["world"]["random_seed"] = 12345
    config["model"]["cnn_channels"] = [16, 32]
    destination = tmp_path / "model.yaml"

    save_config_with_comments(destination, config)

    text = destination.read_text(encoding="utf-8")
    loaded = yaml.safe_load(text)
    assert text.startswith("# 世界设置")
    assert "# 模型结构参数" in text
    assert "random_seed: 12345  # 随机种子" in text
    assert "cnn_channels: [16, 32]  # 卷积层通道数列表" in text
    assert text.index("world:") < text.index("model:") < text.index("agents:")
    assert loaded["world"]["random_seed"] == 12345
    assert loaded["model"]["cnn_channels"] == [16, 32]
