from dataclasses import dataclass
from pathlib import Path

MARKETS = ("WTI", "BRENT", "GASOIL", "HEATOIL", "RBOB", "NATGAS")
HORIZONS = tuple(range(5, 251, 5))

@dataclass(frozen=True)
class ReplicationConfig:
    data_dir: Path = Path("data_required")
    basis: str = "futures_only"
    train_weeks: int = 104
    hidden_factors: int = 3
    first_epochs: int = 1024
    rolling_epochs: int = 36
    learning_rate: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-7
    lambda_w: float = 0.04
    lambda_bias: float = 0.01
    evaluation_start: str = "2015-01-01"
    evaluation_end: str = "2025-12-31"
    seed: int = 7
    expanded: bool = False
