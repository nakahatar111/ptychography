import torch.nn as nn

# Encoder Layers (1000x1000)
def conv(in_channels, out_channels):
  return nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2),
    nn.BatchNorm2d(out_channels),
    nn.GELU()
  )

def conv_max(in_channels, out_channels):
  return nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2),
    nn.MaxPool2d(4, stride=2, padding=1),
    nn.GELU(),
    nn.BatchNorm2d(out_channels)
  )

# Decoder Layers (128x128)
def convTrans(in_channels, out_channels):
  return nn.Sequential(
    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=0),
    nn.BatchNorm2d(out_channels),
    nn.GELU()
  )

class Encoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.block1 = conv_max(1, 8)
    self.block2 = conv(8, 16)
    self.block3 = conv_max(16, 32)
    self.block4 = conv(32, 64)
    self.block5 = conv(64, 128)

  def forward(self, x):
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)
    x = self.block5(x)

    return x

class Decoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.block1 = convTrans(128, 64)
    self.block2 = convTrans(64, 32)
    self.block3 = convTrans(32, 16)
    self.block4 = convTrans(16, 1)
    self.block5 = nn.Conv2d(1, 1, 1, stride=1, padding=1)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)
    x = self.block5(x)
    return self.sigmoid(x)


class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.encoder = Encoder()
    self.phase_decoder = Decoder()
    self.amp_decoder = Decoder()

  def forward(self, diffraction):
    latent_z = self.encoder(diffraction)
    phase = self.phase_decoder(latent_z)
    amp = self.amp_decoder(latent_z)
    return phase, amp