 
import torch
import torch.nn as nn
import torch.nn.functional as F

# Residual CNN Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.skip(x))


# CBAM Attention (Channel + Spatial)
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca
        # Spatial attention
        sa = torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], dim=1)
        sa = self.spatial_attention(sa)
        x = x * sa
        return x


# Fine-Tuned CRNN Model
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()

        # CNN feature extractor with residual blocks
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # (B, 32, 16, 128)

            ResidualBlock(32, 64),
            nn.MaxPool2d(2, 2),  # (B, 64, 8, 64)

            ResidualBlock(64, 128),
            nn.MaxPool2d((2, 1)),  # (B, 128, 4, 64)

            ResidualBlock(128, 256),
            nn.MaxPool2d((2, 1)),  # (B, 256, 2, 64)

            ResidualBlock(256, 512),
            nn.BatchNorm2d(512),
            nn.MaxPool2d((2, 1)),  # (B, 512, 1, 64)
        )

        # CBAM attention
        self.attention = CBAM(512)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.4)

        # Sequence modeling
        self.lstm = nn.LSTM(
            input_size=512, hidden_size=256, num_layers=2, bidirectional=True, batch_first=False
        )
        

        # Output layer
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # CNN backbone
        features = self.cnn(x)
        features = self.attention(features)
        features = self.dropout(features)

        # Reshape for sequence modeling
        b, c, h, w = features.size()
        features = features.squeeze(2)  # (B, C, W)
        features = features.permute(2, 0, 1)  # (W, B, C)

        # LSTM
        recurrent, _ = self.lstm(features)
        recurrent = self.dropout(recurrent)

        # Linear layer
        output = self.fc(recurrent)

        # Log softmax for CTC
        return F.log_softmax(output, dim=2)

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None
        print("Early Stopping created")

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)
