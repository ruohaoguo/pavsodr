from detectron2.config import CfgNode as CN


def add_model_config(cfg):
    # video data
    # DataLoader
    cfg.INPUT.SAMPLING_FRAME_NUM = 2
    cfg.INPUT.SAMPLING_FRAME_RANGE = 5
    cfg.INPUT.SAMPLING_FRAME_SHUFFLE = False
    cfg.INPUT.AUGMENTATIONS = [] # "brightness", "contrast", "saturation", "rotation"

def add_model_config_soundnet(cfg):
    # video data
    # DataLoader
    cfg.INPUT.SAMPLING_FRAME_NUM = 2
    cfg.INPUT.SAMPLING_FRAME_RANGE = 5
    cfg.INPUT.SAMPLING_FRAME_SHUFFLE = False
    cfg.INPUT.AUGMENTATIONS = [] # "brightness", "contrast", "saturation", "rotation"

    # Audio Encoder
    cfg.AUDIO_ENCODER_NAME = "soundnet"
