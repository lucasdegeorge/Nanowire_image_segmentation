#%% 
import torch.nn as nn

img_channels = 2 
noise_channels = 100 # Size of z latent vector (i.e. size of generator input)
gfeatures_size = 64 
dfeatures_size = 64 

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( noise_channels, gfeatures_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(gfeatures_size * 8),
            nn.ReLU(True),
            # state size. ``(gfeatures_size*8) x 4 x 4``
            nn.ConvTranspose2d(gfeatures_size * 8, gfeatures_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gfeatures_size * 4),
            nn.ReLU(True),
            # state size. ``(gfeatures_size*4) x 8 x 8``
            nn.ConvTranspose2d( gfeatures_size * 4, gfeatures_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gfeatures_size * 2),
            nn.ReLU(True),
            # state size. ``(gfeatures_size*2) x 16 x 16``
            nn.ConvTranspose2d( gfeatures_size * 2, gfeatures_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(gfeatures_size),
            nn.ReLU(True),
            # state size. ``(gfeatures_size) x 32 x 32``
            nn.ConvTranspose2d( gfeatures_size, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(img_channels) x 64 x 64``
        )

    def forward(self, input):
        return self.main(input)
    

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(img_channels, dfeatures_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(dfeatures_size) x 32 x 32``
            nn.Conv2d(dfeatures_size, dfeatures_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dfeatures_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(dfeatures_size*2) x 16 x 16``
            nn.Conv2d(dfeatures_size * 2, dfeatures_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dfeatures_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(dfeatures_size*4) x 8 x 8``
            nn.Conv2d(dfeatures_size * 4, dfeatures_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dfeatures_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(dfeatures_size*8) x 4 x 4``
            nn.Conv2d(dfeatures_size * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
    


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)