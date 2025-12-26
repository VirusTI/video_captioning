"""Configuration utilities for Hydra."""

from omegaconf import DictConfig, OmegaConf


def print_config(cfg: DictConfig) -> None:
    """Pretty print configuration.

    Args:
        cfg: Hydra configuration object
    """
    print(OmegaConf.to_yaml(cfg))


def get_config_dict(cfg: DictConfig) -> dict:
    """Convert OmegaConf to dict.

    Args:
        cfg: Hydra configuration object

    Returns:
        Configuration as dictionary
    """
    return OmegaConf.to_container(cfg, resolve=True)


def merge_configs(*configs: DictConfig) -> DictConfig:
    """Merge multiple configurations.

    Args:
        *configs: Variable number of configuration objects

    Returns:
        Merged configuration
    """
    result = OmegaConf.create({})
    for cfg in configs:
        result = OmegaConf.merge(result, cfg)
    return result
