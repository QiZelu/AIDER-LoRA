"""The independent shared-Gate experiment has one pinned configuration."""
from utils.protocol import ROOT, validate_config, validate_data, CONTRACT


def film_video_condition(config):
    return config['model']['film_video_condition']


def validate_model_config(config):
    validate_config(config)


def checkpoint_contract(config):
    validate_config(config)
    return CONTRACT
