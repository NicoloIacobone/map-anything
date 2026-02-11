import logging
import sys
import wandb

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.global_hydra import GlobalHydra

from mapanything.distill.distillation import distillation
from mapanything.utils.misc import StreamToLogger
from hydra.core.global_hydra import GlobalHydra

GlobalHydra.instance().clear()
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="distillation")
def execute_distillation(cfg: DictConfig):
    """
    Execute the distillation process with the provided configuration.

    Args:
        cfg (DictConfig): Configuration object loaded by Hydra
    """
    # Allow the config to be editable
    cfg = OmegaConf.structured(OmegaConf.to_yaml(cfg))

    # Redirect stdout and stderr to the logger
    # sys.stdout = StreamToLogger(log, logging.INFO)
    # sys.stderr = StreamToLogger(log, logging.ERROR)

    # Run the distillation
    distillation(cfg)

if __name__ == "__main__":
    try:
        execute_distillation()
    except KeyboardInterrupt:
        raise
    finally:
        if wandb.run is not None:
            wandb.finish()
