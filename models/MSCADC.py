import torch.nn as nn
from models.CNNBaseModel import CNNBaseModel


class MSCADC(CNNBaseModel):

    """
    MULTI-SCALE CONTEXT AGGREGATION architecture with VGG backend (Large version)
    """

    def __init__(self, num_classes=4, in_channels=1, option='Large', init_weights=True):
        """
        """
        super(MSCADC, self).__init__()


        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = num_classes
        self.middle_channel=32

        if option == 'Large':
            C = [self.middle_channel,2,2,4,8,16,32,32,1]
        elif option == 'Medium':
            C = [self.middle_channel]+[1]*8 + [1]
        else:
            raise ValueError(f"MCSDC does not have {option} version")
        
        for i in range(1,len(C)):
            C[i] *= self.out_channels
        D = [0,1,1,2,4,8,16,1,1]

        self.mscadc = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=C[i-1], out_channels=C[i], 
                            kernel_size=(3,3), dilation=D[i], padding='same'),
                nn.ReLU(inplace=True),
            ) 
            for i in range(1,len(C)-1)
        ]).append(
            nn.Conv2d(in_channels=C[-2], out_channels=C[-1], 
                            kernel_size=(1,1), padding='same')
        )

        # VGG 16 modified
        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, 
                            kernel_size=(3,3), padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, 
                            kernel_size=(3,3), padding='same'),
            nn.ReLU(inplace=True),
        )
        self.second_layer = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, 
                            kernel_size=(3,3), padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, 
                            kernel_size=(3,3), padding='same'),
            nn.ReLU(inplace=True),
        )
        self.third_layer = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=512, 
                            kernel_size=(3,3), padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, 
                            kernel_size=(3,3), padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, 
                            kernel_size=(3,3), padding='same'),
            nn.ReLU(inplace=True),
        )

        self.last_layer = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, 
                            kernel_size=(3,3), padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, 
                            kernel_size=(3,3), padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, 
                            kernel_size=(3,3), padding='same'),
            nn.ReLU(inplace=True),
            # pooling removed, dilatation 2
            nn.Conv2d(in_channels=512, out_channels=512, 
                            kernel_size=(3,3), padding='same', dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, 
                            kernel_size=(3,3), padding='same', dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, 
                            kernel_size=(3,3), padding='same', dilation=2),
            nn.ReLU(inplace=True),
            # pooling removed, dilatation 4
            nn.Conv2d(in_channels=512, out_channels=512, 
                            kernel_size=(3,3), padding='same', dilation=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=128, 
                            kernel_size=(1,1), padding='same', dilation=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=32, 
                            kernel_size=(1,1), padding='same', dilation=4),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2)
        self.up_sample = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.out_channels, out_channels=self.out_channels, 
                                kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=self.out_channels, out_channels=self.out_channels, 
                                kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=self.out_channels, out_channels=self.out_channels, 
                                kernel_size=2, stride=2),
        )


    def forward(self, x):
        """
        Forward pass of the model
        Args:
            x: Tensor
        """
        out = self.first_layer(x)
        out = self.pool(out)
        out = self.second_layer(out)
        out = self.pool(out)
        out = self.third_layer(out)
        out = self.pool(out)
        out = self.last_layer(out)

        for layer in self.mscadc:
            out = layer(out)
        out = self.up_sample(out)
        return out 