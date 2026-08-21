"""Promote an externally evaluated latest checkpoint for tournament inference."""

from .config import BEST_CHECKPOINT_PATH, CHECKPOINT_PATH
from .persistence import atomic_torch_save, load_torch_checkpoint


def main() -> None:
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"No checkpoint to promote at {CHECKPOINT_PATH}")
    atomic_torch_save(load_torch_checkpoint(CHECKPOINT_PATH), BEST_CHECKPOINT_PATH)
    print(f"Promoted {CHECKPOINT_PATH.name} to {BEST_CHECKPOINT_PATH.name}")


if __name__ == "__main__":
    main()
