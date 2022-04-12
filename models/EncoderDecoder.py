import torch.nn as nn
from models.CNNBaseModel import CNNBaseModel


class Encoder_Decoder(CNNBaseModel):

    """
    Achitecture encoder decoder pour la segmentation d'image
    """

    def __init__(self, num_classes=4, in_channels=1, depht=4, init_weights=True):
        """
        depht is the number of pooling (and unpooling) layers
        """
        super(Encoder_Decoder, self).__init__()
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = num_classes
        self.depht=depht
        size = 4

        # couches de l'encodeur
        self.encoder_layers = nn.ModuleList([
            # première couche a une taille d'entrée différente
            nn.Sequential(
                
                nn.Conv2d(in_channels=in_channels, out_channels=size, 
                            kernel_size=3, padding='same'),
                nn.BatchNorm2d(size),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=size, out_channels=size, 
                            kernel_size=3, padding='same'),
                nn.BatchNorm2d(size),
                nn.ReLU(inplace=True),
            )
        # les autres couches on toutes le même combre de carte d'activation
        ] + [ 
            nn.Sequential(
                nn.Conv2d(in_channels=size, out_channels=size, 
                            kernel_size=3, padding='same'),
                nn.BatchNorm2d(size),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=size, out_channels=size, 
                            kernel_size=3, padding='same'),
                            nn.ReLU(inplace=True),
                nn.BatchNorm2d(size),
            )
            for i in range(1,depht)
        ])

        # couches du décodeur
        self.decoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=size, out_channels=size, 
                            kernel_size=3, padding='same'),
                nn.BatchNorm2d(size),
                nn.ReLU(inplace=True),                
                nn.Conv2d(in_channels=size, out_channels=size, 
                            kernel_size=3, padding='same'),
                nn.BatchNorm2d(size),
                nn.ReLU(inplace=True),
            )
            for i in range(depht-1)
        ])
        # la dernière couche à une taille de sortie différente
        self.decoder_layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels=size, out_channels=size, 
                            kernel_size=3, padding='same'),
                nn.BatchNorm2d(size),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=size, out_channels=self.out_channels, 
                            kernel_size=3, padding='same'),
            )
        )

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.max_unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        """
        Forward pass of the model
        Args:
            x: Tensor
        """
        # stacks used to store information
        outputs = [] # remember output sizes for unpooling
        indices = [] # remember pooling indices for the unpooling

        prev_layer = x     
        for layer in self.encoder_layers:
            out = layer(prev_layer) # convolutions 
            prev_layer, ind = self.max_pool(out) # pooling
            indices.append(ind) # save in stacks
            outputs.append(out.size())

        for layer in self.decoder_layers:
            out = self.max_unpool(prev_layer, indices.pop(), output_size=outputs.pop()) # unpooling
            prev_layer = layer(out) # convolutions

        return prev_layer #self.softmax(prev_layer)