import logging
import sys
from os.path import join

from omegaconf import OmegaConf

import wandb
from prompting.misc_utils import seed_everything, setup_logging
from rl import AGENTS, ENVIRONMENTS
from rl.misc_utils import setup_rl_output_dir

logger = logging.getLogger(__name__)


def main():
    base_conf = OmegaConf.load("rl_configs/base.yaml")
    user_conf_files = sys.argv[1:]
    logger.info(f"Reading from the files: {user_conf_files}")

    user_confs = [OmegaConf.load(file) for file in user_conf_files]
    conf = OmegaConf.merge(base_conf, *user_confs)

    setup_rl_output_dir(conf)
    setup_logging(conf)

    if conf.output_dir:
        with open(join(conf.output_dir, "config.yaml"), "w") as f:
            OmegaConf.save(config=conf, f=f)

    if conf.get("wandb_resume") and conf.get("group") and conf.get("name"):
        wandb_id = f"{conf['group']}.{conf['name']}"
    else:
        wandb_id = wandb.util.generate_id()

    wandb.init(
        config=conf,
        project="active-example-selection",
        name=conf.get("name"),
        resume="allow",
        group=conf.get("group"),
        id=wandb_id,
    )

    logger.info(str(conf))
    seed = conf.get("seed", 42)
    logger.info(f"Seed: {seed}")
    seed_everything(seed)

    env_type = ENVIRONMENTS[conf.env]
    env = env_type(seed=seed, **conf.env_kwargs)
    agent_type = AGENTS[conf.agent]
    agent = agent_type(env, conf.output_dir, **conf.agent_kwargs)

    agent.train()

    for test_kwargs in conf.tests:
        agent.eval(eval_mode="test", **test_kwargs)
    del agent

    for test in conf.full_tests:
        if "env_kwargs" in test:
            test_env_kwargs = OmegaConf.merge(conf.env_kwargs, test.env_kwargs)
        else:
            test_env_kwargs = conf.env_kwargs
        test_env = env_type(seed=seed, **test_env_kwargs)
        test_agent = agent_type(test_env, conf.output_dir, **conf.agent_kwargs)
        test_agent.eval(eval_mode="test", **test.test_kwargs)

    logger.info("Done!")


if __name__ == "__main__":
    main()
