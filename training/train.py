import hydra
from hydra.utils import instantiate


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="config"
)
def main(cfg):

    trainer = instantiate(cfg.model)

    trainer.train()


if __name__ == "__main__":
    main()