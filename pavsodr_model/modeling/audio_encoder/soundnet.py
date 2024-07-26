import torch
import torch.nn as nn


class SoundNet(nn.Module):
    def __init__(self):
        super(SoundNet, self).__init__()

        self.soundnet = nn.Sequential(
            nn.Conv2d(1, 16, (1, 64), (1, 2), (0, 32)), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d((1, 8), (1, 8)),
            nn.Conv2d(16, 32, (1, 32), (1, 2), (0, 16)), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((1, 8), (1, 8)),
            nn.Conv2d(32, 64, (1, 16), (1, 2), (0, 8)), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, (1, 8), (1, 2), (0, 4)), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, (1, 4), (1, 2), (0, 2)), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d((1, 4), (1, 4)),
            nn.Conv2d(256, 512, (1, 4), (1, 2), (0, 2)), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 1024, (1, 4), (1, 2), (0, 2)), nn.BatchNorm2d(1024), nn.ReLU(), nn.MaxPool2d((1, 2))
        )

        # load pre-training model
        self.initialize_soundnet()

        # freeze audio branch
        for param in self.soundnet.parameters():
            param.requires_grad = False


    def forward(self, x):
        with torch.no_grad():
            feats_audio = self.soundnet(x.unsqueeze(1))
            if feats_audio.shape[0] == 1:
                feats_audio = torch.mean(feats_audio, dim=-1).squeeze().unsqueeze(0)
            else:
                feats_audio = torch.mean(feats_audio, dim=-1).squeeze()

            return feats_audio


    def initialize_soundnet(self):
        audio_pretrain_weights = torch.load("./pre_models/soundnet8.pth")

        all_params = {}
        for k, v in self.soundnet.state_dict().items():
            if 'module.soundnet8.' + k in audio_pretrain_weights.keys():
                v = audio_pretrain_weights['module.soundnet8.' + k]
                all_params[k] = v
        self.soundnet.load_state_dict(all_params)

