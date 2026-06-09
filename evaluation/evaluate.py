import hydra
from hydra.utils import instantiate


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="config"
)
def main(cfg):

    evaluator = instantiate(
        cfg.evaluator
    )

    evaluator.evaluate(
        cfg.dataset
    )


if __name__ == "__main__":
    main()