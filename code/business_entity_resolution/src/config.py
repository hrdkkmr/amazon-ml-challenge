"""Central configuration: paths and tunable parameters.

Values can be overridden with a JSON file passed as ``--config`` to
``pipeline.py`` (see configs/default.json).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    data_dir: str = str(ROOT / "dataset")
    cache_dir: str = str(ROOT / "cache")
    model_dir: str = str(ROOT / "models")
    output_dir: str = str(ROOT / "output")
    experiments_dir: str = str(ROOT / "experiments")

    # preprocessing
    read_chunk: int = 200_000
    n_workers: int = max(1, min(8, (os.cpu_count() or 2) - 2))

    # blocking
    df_cap_name: int = 3000       # keys occurring in more target records are ignored
    df_cap_addr: int = 3000
    query_chunk: int = 20_000     # S1 rows per sparse-product batch
    top_k: int = 12               # candidates kept per S1 per target source
    min_block_score: float = 0.0

    # validation split
    valid_frac: float = 0.2
    seed: int = 42

    # training
    neg_per_s1: int = 0           # 0 = keep every candidate negative
    model: str = "hgb"

    # decision
    threshold: float = 0.5
    one_to_one: bool = True

    extra: dict = field(default_factory=dict)

    def path(self, *parts) -> Path:
        return Path(self.cache_dir).joinpath(*parts)

    @classmethod
    def load(cls, path: str | None = None) -> "Config":
        cfg = cls()
        if path:
            for k, v in json.loads(Path(path).read_text()).items():
                setattr(cfg, k, v)
        for d in (cfg.cache_dir, cfg.model_dir, cfg.output_dir, cfg.experiments_dir):
            Path(d).mkdir(parents=True, exist_ok=True)
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)
