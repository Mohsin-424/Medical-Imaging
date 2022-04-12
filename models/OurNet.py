
import torch.nn as nn
import torch 
from models.CNNBaseModel import CNNBaseModel
import warnings



class OurNet(CNNBaseModel):

    """
    Achitecture personnalisé entre ResNet et UNet
    """

    def __init__(self, num_classes=4, in_channels=1, init_weights=True, depth=4, option="Large"):
        """
        
        """
        super(OurNet, self).__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = num_classes

        size=64

        #gestion de l'hyper paramêtre du nombre de couches et de la profondeur
        self.layer_size=calcul_layer_size(depth,option)
        t=len(self.layer_size)
        self.reverse_layer_size=list(reversed(self.layer_size[0:(t-1)]))
        #première couche 
        self.first_layer=nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        #descente en enfer
        self.down_layer=nn.ModuleList()

        for j in range(0,depth):

                # couche de convolution avec une stride de 2 pour la descente:   
                self.down_layer.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels=size*2**j, out_channels=size*2**(j+1), kernel_size=2, stride=2),
                        nn.BatchNorm2d(size*2**(j+1)),
                        nn.ReLU(inplace=True),
                    )
                )
         
                # couches de bottleneck par étage:
                for i in range(0,self.layer_size[j]):
                    self.down_layer.append(
                        nn.Sequential(
                            nn.Conv2d(in_channels=size*2**(j+1), out_channels=int(size*2**(j-1)), kernel_size=1, stride=1, padding="same"),
                            nn.BatchNorm2d(int(size*2**(j-1))),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(in_channels=int(size*2**(j-1)), out_channels=int(size*2**(j-1)), kernel_size=3, stride=1, padding="same"),
                            nn.BatchNorm2d(int(size*2**(j-1))),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(in_channels=int(size*2**(j-1)), out_channels=size*2**(j+1), kernel_size=1, stride=1, padding="same"),
                            nn.BatchNorm2d(size*2**(j+1)),
                        )
                    )

        self.first_unconv=nn.Sequential(
            nn.ConvTranspose2d(in_channels=size*2**depth, out_channels=int(size*2**(depth-1)), kernel_size=2, stride=2),
            nn.BatchNorm2d(int(size*2**(depth-1))),
            nn.ReLU(inplace=True),
        )


        # remonter vers les nuages
        self.up_layer=nn.ModuleList()
        for j in range(depth-1,0,-1):  

            self.up_layer.append(
                # couche de convolution pour diminuer le nombre de channels après la concaténation
                nn.Sequential(
                    nn.Conv2d(in_channels=size*2**(j+1), out_channels=size*2**j, kernel_size=1, stride=1, padding="same"),
                    nn.BatchNorm2d(size*2**j),
                    nn.ReLU(inplace=True),
                )
            )

            for i in range(0,self.reverse_layer_size[len(self.reverse_layer_size)-j]):

                self.up_layer.append(
                    # couches de bottlenecks 
                    nn.Sequential(
                        nn.Conv2d(in_channels=size*2**j, out_channels=int(size*2**(j-2)), kernel_size=1, stride=1, padding="same"),
                        nn.BatchNorm2d(int(size*2**(j-2))),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(in_channels=int(size*2**(j-2)), out_channels=int(size*2**(j-2)), kernel_size=3, stride=1, padding="same"),
                        nn.BatchNorm2d(int(size*2**(j-2))),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(in_channels=int(size*2**(j-2)), out_channels=size*2**j, kernel_size=1, stride=1, padding="same"),
                        nn.BatchNorm2d(size*2**j),
                    )
                )
            
            # couche de déconvolution avec une stride de 2 pour la descente
            self.up_layer.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels=size*2**j, out_channels=int(size*2**(j-1)), kernel_size=2, stride=2),
                    nn.BatchNorm2d(int(size*2**(j-1))),
                    nn.ReLU(inplace=True),
                )
            )

        self.last_layer=nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=self.num_classes, kernel_size=3, stride=1, padding="same"),
        )

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        """
        Forward pass of the model
        Args:
            x: Tensor
        """
        layer_outputs=[]
        bottleneck_output=0

        out=self.first_layer(x)
        

        down_index=[sum(self.layer_size[0:i])+i for i in range(len(self.layer_size))]
        count=0
        for layer in self.down_layer:
            if(count in down_index):
                #la couche qui sort est une couche de convolution stride 2 pour descendre
                #donc on sauvegarder l'image avant de convoluer pour la réutiliser
                layer_outputs.append(out)
                out=layer(out)
                bottleneck_output=out
                
            else:
                #la couche qui sort est un bottleneck donc on effectue une somme avec la sortie précédente puis un relu
                out=layer(out)
                out+=bottleneck_output
                out=self.relu(out)
                bottleneck_output=out

            count+=1
        
        out = self.first_unconv(out)
        out = torch.cat((out,layer_outputs.pop()),dim=1)
        

        # calcul des indices des différentes couches dans la liste
        t=len(self.layer_size)
        
        up_index_trans=[sum(self.reverse_layer_size[0:i])+2*i-1 for i in range(1,len(self.layer_size))]
        up_index_red=[sum(self.reverse_layer_size[0:i])+2*i for i in range(0,len(self.layer_size)-1)]
        count=0
        for layer in self.up_layer:
            if(count in up_index_red):
                #couche de redimentionnement
                out=layer(out)
                bottleneck_output=out
            elif(count in up_index_trans):
                #couche de convTranspose
                out=layer(out)
                out = torch.cat((out,layer_outputs.pop()),dim=1)

            else:
                #couche de bottleneck
                out=layer(out)
                out+=bottleneck_output
                out=self.relu(out)
                bottleneck_output=out
            count+=1
        
        out = self.last_layer(out)

        return out
    
#calcul à partir des options la largeur de toutes les couches
def calcul_layer_size(depth,option):
    if (option=="Small"):
        incr=[0 for i  in range(depth)]
    elif (option=="Medium"):
        incr=[2 for i in range(depth)]
        incr[0]=1
        incr[depth-1]=1
    elif (option=="Large"):
        incr=[3 for i in range(depth)]
        incr[depth-1]=1
        if(depth>=3):
            incr[1]=2
    else :
        warnings.warn("Error no valide option for ResNet, using default configuration")
        incr=[0 for i  in range(depth)]
    result=[i+incr[i] for i in range(depth)]
    return result

