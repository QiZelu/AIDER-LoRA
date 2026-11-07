"""Model factory for the independent ordinary Transformer/task-QK-gate experiment."""

from .registry import TASKS, BASELINE_NAME, BASELINE_VARIANT


def build_aider(config, build_backbones=True):
    if (
        config['model']['name'] != BASELINE_NAME
        or config['model'].get('readout_variant') != BASELINE_VARIANT
        or (tuple(config['task']['order']) != TASKS and 'videofeedback' not in config)
    ):
        raise ValueError('Expected the four-task Transformer/QK-gate configuration')
    from .support import seed_everything
    from .aider import AIDER

    seed_everything(config['training']['seed'])
    return AIDER(config, build_backbones=build_backbones)


def __getattr__(name):
    if name == 'AIDER':
        from .aider import AIDER

        return AIDER
    raise AttributeError(name)


from .support import load_config

__all__ = ['AIDER', 'build_aider', 'load_config', 'TASKS']
