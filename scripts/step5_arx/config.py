"""Pinned model and optimizer settings for the separate ARX experiment."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

OPENPI = "f4e52e6de4cc487a11585a3a7414d610b2883dae"
PACE_MODEL = "17ef545270cf26ed81169893b60f845f80ce6557"
CODE_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = CODE_ROOT / "pace_download_manifest.json"


@dataclass(frozen=True)
class Settings:
    config_id: str = "pi05_robotwin_handover_block_arx_x5_lora"
    seed: int = 42
    paligemma_variant: str = "gemma_2b_lora"
    action_expert_variant: str = "gemma_300m_lora"
    learning_rate: float = 3e-5
    gradient_clip: float = 1.0
    weight_decay: float = 0.0
    batch_size: int = 1
    train_steps: int = 400
    checkpoint_interval: int = 25
    validation_interval: int = 100
    evaluation_stride: int = 25
    denoise_steps: int = 10
    overfit_steps: int = 20
    overfit_min_relative_improvement: float = 0.01
    ema: bool = False

    def as_dict(self):
        return asdict(self)


SETTINGS = Settings()


def write_json(path, values):
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(values, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)
