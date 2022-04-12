import torch.nn as nn

from models.CNN import CNNet
from models.EncoderDecoder import Encoder_Decoder
from models.UNet_enhance import UNet
from models.OurNet import OurNet
from models.MSCADC import MSCADC

class ModelFactory(nn.Module):
    '''
    Class to handle model creation and their hyperparameters
    Using Factory design pattern
    '''

    # Model corresponding to model names:
    model_dict = {
                "CNNet": CNNet,
                "ENC":Encoder_Decoder,
                "UNet": UNet,
                "OurNet": OurNet,
                "MSCADC": MSCADC,
    }

    # Returns the factory corresponding to the model_name
    def getFactory(model_name = "CNNet"):
        if ((model_name) == list(ModelFactory.model_dict)[0]):
            return CNNetFactory()
        elif ((model_name) == list(ModelFactory.model_dict)[1]):
            return EncDecFactory()
        elif ((model_name) == list(ModelFactory.model_dict)[2]):
            return UNetFactory()          
        elif ((model_name) == list(ModelFactory.model_dict)[3]):
            return OurNetFactory()            
        elif ((model_name) == list(ModelFactory.model_dict)[4]):
            return MscadcFactory()
        
        # When overriding this function, one must use the same parameters name. If a new parameter is used, it must be added here
        def getModel(self, num_classes, in_channels, depth, option, *args, **kwargs):
            return NotImplementedError

    def has_depth(self):
        return False

    def has_option(self):
        return False

class CNNetFactory(ModelFactory):
    # simply allow additional args
    def getModel(self, num_classes, in_channels, *args, **kwargs):
        return CNNet(num_classes, in_channels)

class EncDecFactory(ModelFactory):
    def getModel(self, num_classes, in_channels, depth, *args, **kwargs):
        return Encoder_Decoder(num_classes, in_channels, depth)

    def has_depth(self):
        return True

class UNetFactory(ModelFactory):
    def getModel(self, num_classes, in_channels, depth, *args, **kwargs):
        return UNet(num_classes, in_channels, depth)

    def has_depth(self):
        return True

class MscadcFactory(ModelFactory):
    def getModel(self, num_classes, in_channels, option, *args, **kwargs):
        return MSCADC(num_classes, in_channels, option=option)

    def has_option(self):
        return True    

class OurNetFactory(ModelFactory):
    def getModel(self, num_classes, in_channels, depth, option, *args, **kwargs):
        return OurNet(num_classes, in_channels, depth=depth, option=option)

    def has_option(self):
        return True   
    
    def has_depth(self):
        return True


    
        

        