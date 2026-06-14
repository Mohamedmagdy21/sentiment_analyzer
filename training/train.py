import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

import hydra
from hydra.utils import instantiate


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="config"
)
def main(cfg):

    trainer = instantiate(cfg.model)

    trainer.train(cfg.dataset)


if __name__ == "__main__":
    main()