import torch
from torch import nn





class GAN(nn.Moudle):
    class _Generator(nn.Module):
        def __init__(self, 
                     intermediate_channels,
                     image_channels,):
            self.out_channels = image_channels
            self.intermediate_channels = intermediate_channels
            super(GAN._Generator, self).__init__()
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d( in_channels=1, out_channels=self.intermediate_channels * 8, kernel_size=4, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.intermediate_channels * 8),
                nn.ReLU(True),
                # state size. ``(intermediate_channels*8) x 4 x 4``
                nn.ConvTranspose2d(self.intermediate_channels * 8, self.intermediate_channels * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.intermediate_channels * 4),
                nn.ReLU(True),
                # state size. ``(intermediate_channels*4) x 8 x 8``
                nn.ConvTranspose2d( self.intermediate_channels * 4, self.intermediate_channels * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.intermediate_channels * 2),
                nn.ReLU(True),
                # state size. ``(intermediate_channels*2) x 16 x 16``
                nn.ConvTranspose2d( self.intermediate_channels * 2, self.intermediate_channels, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.intermediate_channels),
                nn.ReLU(True),
                # state size. ``(intermediate_channels) x 32 x 32``
                nn.ConvTranspose2d(self.intermediate_channels, self.out_channels, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. ``(nc) x 64 x 64``
            )

        def forward(self, input: torch.Tensor):
            input = input.view(10,10)
            return self.main(input)

    class _Discriminator(nn.Module):
        def __init__(self,
                     image_channels : int = 3,
                     intermediate_channels : int = 1
                     ):
            self.image_channels = image_channels
            self.intermediate_channels = intermediate_channels
            super(GAN._Discriminator, self).__init__()
            self.main = nn.Sequential(
                # input is ``(nc) x 64 x 64``
                nn.Conv2d(self.image_channels, self.intermediate_channels, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. ``(ndf) x 32 x 32``
                nn.Conv2d(self.intermediate_channels, self.intermediate_channels * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.intermediate_channels * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. ``(ndf*2) x 16 x 16``
                nn.Conv2d(self.intermediate_channels * 2, self.intermediate_channels * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.intermediate_channels * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. ``(ndf*4) x 8 x 8``
                nn.Conv2d(self.intermediate_channels * 4, self.intermediate_channels * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.intermediate_channels * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. ``(ndf*8) x 4 x 4``
                nn.Conv2d(self.intermediate_channels * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

        def forward(self, input : torch.Tensor):
            return self.main(input)


    def __init__(self,
                 latent_vector_size,
                 generated_image_channels
                 ):
        super.__init__(GAN, self)
        self.latent_vector_size = latent_vector_size
        self.generated_image_channels =generated_image_channels
        self.discriminator = self._Discriminator()
        self.generator = self._Generator()


gan_instance = GAN()

gan_instance.discriminator