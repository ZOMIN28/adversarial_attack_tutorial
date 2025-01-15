import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)

class Generator(nn.Module):
    def __init__(self,in_dim=3, out_dim=64):
        super(Generator, self).__init__()
        # encoder
        self.en_layer1 = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01)
        )
        self.en_layer2 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim*2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim*2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01)
        )
        self.en_layer3 = nn.Sequential(
            nn.Conv2d(out_dim*2, out_dim*4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim*4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01)
        )
        self.en_layer4 = nn.Sequential(
            nn.Conv2d(out_dim*4, out_dim*8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim*8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01)
        )
        self.en_layer5 = nn.Sequential(
            nn.Conv2d(out_dim*8, out_dim*16, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim*16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01)
        )
        # residual block
        self.Residual = nn.Sequential(
            ResidualBlock(dim_in=out_dim*16, dim_out=out_dim*16),
            ResidualBlock(dim_in=out_dim*16, dim_out=out_dim*16)
        )
        # decoder
        self.de_layer1 = nn.Sequential(
            nn.ConvTranspose2d(out_dim*16, out_dim*8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim*8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.de_layer2 = nn.Sequential(
            nn.ConvTranspose2d(out_dim*8, out_dim*4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim*4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.de_layer3 = nn.Sequential(
            nn.ConvTranspose2d(out_dim*4, out_dim*2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim*2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.de_layer4 = nn.Sequential(
            nn.ConvTranspose2d(out_dim*2, out_dim, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
        )
        self.de_layer5 = nn.Sequential(
            nn.ConvTranspose2d(out_dim, in_dim, 4, stride=2, padding=1),
            nn.Tanh()
        )


    def forward(self,x):
        x = self.en_layer5(self.en_layer4(self.en_layer3(self.en_layer2(self.en_layer1(x)))))
        z = self.Residual(x)
        y = self.de_layer5(self.de_layer4(self.de_layer3(self.de_layer2(self.de_layer1(z))))) 
        return y

    


            