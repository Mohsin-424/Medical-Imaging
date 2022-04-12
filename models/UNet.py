import torch.nn as nn
import torch 
from models.CNNBaseModel import CNNBaseModel



class UNet(CNNBaseModel):

    """
    Première version de UNet implémentée, n'est plus utilisation dans la ligne de commande, car remplacé par le fichier UNet_enhance.py
    """

    def __init__(self, num_classes=4, in_channels=1, init_weights=True):
        """
        
        """
        super(UNet, self).__init__()
        # encoder

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = num_classes


        # fonction d'activation
        self.relu = nn.ReLU()


        # partie descendante: couches de convolution
        # stride = 1
        # noyau = 3*3*3
        # padding = 0 , pas de padding

        self.conv_layers1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(64),
        )
        
        self.conv_layers2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(128),
        )

        self.conv_layers3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(256),
        )

        self.conv_layers4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(512),
        )

        self.conv_layers5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(512),
        )


        # partie descendante: couches de max pooling
        # stride = 2
        # noyau = 2*2*2
        # padding = 0, pas de padding
        # à utiliser 4 fois

        self.maxPool=nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)


        # partie montante: couches de convolution
        # stride = 1
        # noyau = 3*3*3
        # padding = 0 , pas de padding

        self.conv_layers6 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(256),
        )
        
        self.conv_layers7 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(128),
        )

        self.conv_layers8 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(64),
        )

        self.conv_layers9 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=3, stride=1, padding="same"),
        )

        # partie montante: couches inverses de max pooling
        # stride = 2
        # noyau = 2*2*2
        # padding = 0, pas de padding
        # à utiliser 4 fois

        self.maxUnPool = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)


    def forward(self, x):
        """
        Forward pass of the model
        Args:
            x: Tensor
        """

        # partie encoder
        # etage 1
        output = self.relu(self.conv_layers1(x))
        copy_stage1 = output

        # descente 1
        output,indices_stage1=self.maxPool(output)

        # etage 2
        output = self.relu(self.conv_layers2(output))
        copy_stage2 = output

        # descente 2
        output,indices_stage2=self.maxPool(output)

        # etage 3
        output = self.relu(self.conv_layers3(output))
        copy_stage3 = output

        # descente 3
        output,indices_stage3=self.maxPool(output)

        # etage 4
        output = self.relu(self.conv_layers4(output))
        copy_stage4 = output

        # descente 4
        output,indices_stage4=self.maxPool(output)

        # etage 5
        output = self.relu(self.conv_layers5(output))
       
        # vérifications :
        #print("")
        #print("encodeur")
        #print("fin etage 1 -> [5,64,256,256]")
        #print(copy_stage1.size())
        #print("fin étage 2 -> [5,128,128,128]")
        #print(copy_stage2.size())
        #print("fin étage 3 -> [5,256,64,64]")
        #print(copy_stage3.size())
        #print("fin étage 4 -> [5,512,32,32]")
        #print(copy_stage4.size())
        #print("fin étage 5 -> [5,1024,16,16]")
        #print(output.size())
        #print("")

        # partie decoder
        # remonter 1
        #print(indices_stage4.size())
        #print(copy_stage4.size())
        output = self.maxUnPool(output,indices_stage4,output_size=copy_stage4.size())
        # vérification :
        #print("")
        #print("decodeur")
        #print("début etage 4 -> [5,512,32,32]")
        #print(output.size())

        # etage 4
        output = torch.cat((output,copy_stage4),dim=1)
        output = self.relu(self.conv_layers6(output))
        #print("fin etage 4 -> [5,512,32,32]")
        #print(output.size())

        #remonter 2
        #print(indices_stage3.size())
        #print(copy_stage3.size())
        output = self.maxUnPool(output,indices_stage3,output_size=copy_stage3.size())
        #print("début etage 3 -> [5,256,64,64]")
        #print(output.size())

        # etage 3
        output = torch.cat((output,copy_stage3),dim=1)
        output = self.relu(self.conv_layers7(output))
        #print("fin etage 3 -> [5,256,64,64]")
        #print(output.size())

        # remonter 3
        output = self.maxUnPool(output,indices_stage2,output_size=copy_stage2.size())
        #print("début etage 2 -> [5,128,128,128]")
        #print(output.size())

        # etage 2
        output = torch.cat((output,copy_stage2),dim=1)
        output = self.relu(self.conv_layers8(output))
        #print("fin etage 2 -> [5,128,256,256]")
        #print(output.size())

        #remonter 4
        output = self.maxUnPool(output,indices_stage1,output_size=copy_stage1.size())
        #print("début etage 1 -> [5,64,512,512]")
        #print(output.size())
        # etage 1
        output = torch.cat((output,copy_stage1),dim=1)
        output = self.conv_layers9(output)
        #print("fin etage 1 -> [5,",self.num_classes,",512,512]")
        #print(output.size())

        return output