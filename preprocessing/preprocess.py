# preprocessing/preprocess.py
import hydra
from hydra.utils import instantiate


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="config"
)
def main(cfg):
    """Instantiate the preprocessor from Hydra config and run text cleaning, label mapping, and train/val/test split."""
    processor = instantiate(
        cfg.preprocessing
    )

    processor.run(
        cfg.dataset
    )


if __name__ == "__main__":
    main()