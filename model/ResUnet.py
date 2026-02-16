import torch 
import torch.nn as nn 


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool = True):
        super().__init__()
        self.downsample = downsample
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.res_block = ResidualBlock(out_channels)
        
        if downsample:
            self.pool = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                                 stride=2, padding=1, bias=False)
    
    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.res_block(x)
        
        if self.downsample:
            skip = x
            x = self.pool(x)
            return x, skip
        else:
            return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels + skip_channels, out_channels, 
                             kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.res_block = ResidualBlock(out_channels)
    
    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn(self.conv(x)))
        x = self.res_block(x)
        return x

class ResUNetAutoencoder(nn.Module):
    def __init__(self, in_channels: int = 7, latent_dim: int = 256):
        super().__init__()
        
        # Encoder
        self.enc1 = EncoderBlock(in_channels, 32, downsample=True)
        self.enc2 = EncoderBlock(32, 64, downsample=True)
        self.enc3 = EncoderBlock(64, 128, downsample=True)
        self.bottleneck = EncoderBlock(128, 256, downsample=False)
        
        # Latent
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.latent_dim = latent_dim
        
        # Decoder
        self.dec1 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec3 = DecoderBlock(64, 32, 32)
        self.final_conv = nn.Conv2d(32, in_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # Encoder
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x = self.bottleneck(x)
        
        # Latent
        latent = self.global_pool(x)
        latent = latent.view(latent.size(0), -1)
        
        # Decoder
        x = self.dec1(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec3(x, skip1)
        
        # Final Conv (Reduces 32 channels -> 7 channels)
        x = self.final_conv(x)
        reconstruction = torch.sigmoid(x)
        return reconstruction, latent
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
class ResUNetEncoder(nn.Module):
    """Encoder-only version for embedding extraction"""
    def __init__(self, in_channels: int = 7, embedding_dim: int = 128):
        super().__init__()
        
        # Encoder
        self.enc1 = EncoderBlock(in_channels, 32, downsample=True)
        self.enc2 = EncoderBlock(32, 64, downsample=True)
        self.enc3 = EncoderBlock(64, 128, downsample=True)
        self.bottleneck = EncoderBlock(128, 256, downsample=False)
        
        # Embedding projection
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.embedding_dim = embedding_dim
        self.projection = nn.Linear(256, embedding_dim)
    
    def forward(self, x):
        # Encoder path
        x, _ = self.enc1(x)
        x, _ = self.enc2(x)
        x, _ = self.enc3(x)
        x = self.bottleneck(x)
        
        # Global pooling and projection
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        embedding = self.projection(x)
        
        return embedding
    
    def load_from_autoencoder(self, autoencoder_path: str):
        """Load encoder weights from trained autoencoder"""
        state_dict = torch.load(autoencoder_path, map_location='cpu')
        
        # Filter encoder weights only
        encoder_dict = {k: v for k, v in state_dict.items() 
                       if k.startswith(('enc', 'bottleneck', 'global_pool'))}
        
        # Load compatible weights
        self.load_state_dict(encoder_dict, strict=False)
        print(f"✓ Loaded encoder weights from {autoencoder_path}")