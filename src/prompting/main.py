import logging
import sys

from omegaconf import OmegaConf

from prompting import MODELS, PROCESSORS, STRATEGIES, PromptTooLongError
from prompting.misc_utils import seed_everything, setup_logging, setup_output_dir

logger = logging.getLogger(__name__)


def main():

    base_conf = OmegaConf.load("prompting_configs/base.yaml")

    user_conf_files = sys.argv[1:]
    logger.info(f"Reading from the files: {user_conf_files}")

    user_confs = [OmegaConf.load(file) for file in user_conf_files]
    conf = OmegaConf.merge(base_conf, *user_confs)
    setup_output_dir(conf)
    setup_logging(conf)

    logger.info(str(conf))

    seed = conf.seed
    logger.info(f"Seed: {seed}")
    seed_everything(seed)

    processor = PROCESSORS[conf.task](seed=seed, **conf.processor_kwargs)
    model_type = MODELS[conf.model]
    model = model_type(conf.model_name, **processor.model_kwargs, **conf.model_kwargs)
    strategy = STRATEGIES[conf.strategy](conf)

    conf.shots = sorted(conf.shots)  # work from small to large shots

    for shot in conf.shots:
        logger.info(f"start working on shot k={shot}")
        try:
            results = strategy(processor, model, shot)
        except PromptTooLongError as e:
            logger.warning(f"caught PromptTooLongError, msg: {e}")
            logger.warning(f"max feasible shot = {shot-1}, exits gracefully...")
            exit(0)
    logging.info("Done!")


if __name__ == "__main__":
    main()
