"""Unified model architectures: SSLClassifier, 2D-CNN (Mel), Bi-LSTM (MFCC), and parameter counting."""

import torch
import torch.nn as nn
import torchaudio
from transformers import AutoModel


class SSLClassifier(nn.Module):
    """
    Unified self-supervised speech emotion classifier:
    Mean-pooling over temporal hidden states followed by a single Linear(768, 8) head.

    Modes:
      - 'full_ft':      Unfreeze CNN feature extractor + Transformer encoder + classifier
      - 'partial_ft':   Freeze CNN feature extractor; train Transformer encoder + classifier
      - 'linear_probe': Freeze entire SSL backbone; train only Linear(768, 8)
      - 'weighted_sum': SUPERB-style learnable softmax-weighted sum over all 13 hidden states + Linear(768, 8)
    """

    def __init__(
        self,
        model_name="facebook/wav2vec2-base",
        num_classes=8,
        mode="full_ft",
        hidden_dropout=0.3,
        attention_dropout=0.3,
        local_files_only=False,
    ):
        super().__init__()
        self.model_name = model_name
        self.mode = mode
        self.num_classes = num_classes

        if mode in ("linear_probe", "weighted_sum"):
            self.backbone = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        else:
            self.backbone = AutoModel.from_pretrained(
                model_name,
                hidden_dropout=hidden_dropout,
                attention_dropout=attention_dropout,
                local_files_only=local_files_only,
            )

        if hasattr(self.backbone, "masked_spec_embed") and self.backbone.masked_spec_embed is not None:
            self.backbone.masked_spec_embed.requires_grad = False

        hidden_size = self.backbone.config.hidden_size
        num_layers = self.backbone.config.num_hidden_layers + 1

        if mode == "weighted_sum":
            self.layer_weights = nn.Parameter(torch.zeros(num_layers))
        else:
            self.register_parameter("layer_weights", None)

        self.classifier = nn.Linear(hidden_size, num_classes)
        self._configure_freeze()

    def _configure_freeze(self):
        if self.mode == "full_ft":
            for p in self.backbone.parameters():
                p.requires_grad = True
            if hasattr(self.backbone, "masked_spec_embed") and self.backbone.masked_spec_embed is not None:
                self.backbone.masked_spec_embed.requires_grad = False
        elif self.mode == "partial_ft":
            for p in self.backbone.parameters():
                p.requires_grad = True
            self.backbone.feature_extractor.requires_grad_(False)
            if hasattr(self.backbone, "masked_spec_embed") and self.backbone.masked_spec_embed is not None:
                self.backbone.masked_spec_embed.requires_grad = False
        elif self.mode in ("linear_probe", "weighted_sum"):
            self.backbone.requires_grad_(False)
        else:
            raise ValueError(f"Unknown SSLClassifier mode: {self.mode}")

        for p in self.classifier.parameters():
            p.requires_grad = True

    def forward(self, input_values, attention_mask=None):
        if self.mode == "weighted_sum":
            with torch.no_grad():
                outputs = self.backbone(
                    input_values=input_values,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
            hidden_states = torch.stack(outputs.hidden_states, dim=0)
            norm_weights = torch.softmax(self.layer_weights, dim=0).view(-1, 1, 1, 1)
            pooled = (hidden_states * norm_weights).sum(dim=0).mean(dim=1)
        elif self.mode == "linear_probe":
            with torch.no_grad():
                outputs = self.backbone(input_values=input_values, attention_mask=attention_mask)
            pooled = outputs.last_hidden_state.mean(dim=1)
        else:
            outputs = self.backbone(input_values=input_values, attention_mask=attention_mask)
            pooled = outputs.last_hidden_state.mean(dim=1)

        return self.classifier(pooled)


class EmotionCNN(nn.Module):
    """4-layer 2D-CNN baseline over 64-bin log-Mel spectrograms."""

    def __init__(self, num_classes=8):
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_mels=64, n_fft=1024, hop_length=512
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, input_values, attention_mask=None):
        x = self.melspec(input_values)
        x = self.amplitude_to_db(x).unsqueeze(1)
        x = self.conv_blocks(x)
        return self.fc(x.view(x.size(0), -1))


class EmotionLSTM(nn.Module):
    """2-layer Bidirectional LSTM baseline over 40-dimensional MFCC features."""

    def __init__(self, num_classes=8):
        super().__init__()
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=16000,
            n_mfcc=40,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 64},
        )
        self.lstm = nn.LSTM(
            input_size=40,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.fc = nn.Linear(128 * 2, num_classes)

    def forward(self, input_values, attention_mask=None):
        x = self.mfcc(input_values).permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out.mean(dim=1))


def count_params(model):
    """Return total and trainable parameter counts (excluding unused masked_spec_embed)."""
    total = sum(p.numel() for n, p in model.named_parameters() if "masked_spec_embed" not in n)
    trainable = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "masked_spec_embed" not in n)
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "total_M": round(total / 1e6, 2),
        "trainable_M": round(trainable / 1e6, 2),
        "trainable_K": round(trainable / 1e3, 2),
    }


def build_model_from_config(cfg, local_files_only=False):
    """Instantiate model and normalization flag from a YAML config dict."""
    mtype = cfg["model_type"]
    if mtype == "cnn":
        return EmotionCNN(num_classes=cfg.get("num_classes", 8)), cfg.get("normalize_waveform", False)
    if mtype == "lstm":
        return EmotionLSTM(num_classes=cfg.get("num_classes", 8)), cfg.get("normalize_waveform", False)
    if mtype in ("wav2vec2", "wavlm"):
        model = SSLClassifier(
            model_name=cfg["backbone"],
            num_classes=cfg.get("num_classes", 8),
            mode=cfg["mode"],
            local_files_only=local_files_only,
        )
        return model, cfg.get("normalize_waveform", True)
    raise ValueError(f"Unknown model_type: {mtype}")
