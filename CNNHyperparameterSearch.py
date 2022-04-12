from sklearn.metrics import det_curve
import torch.optim as optim
import torch.nn as nn

from os.path import join

from CNNTrainTestManager import CNNTrainTestManager, optimizer_setup
from models.CNNBaseModel import CNNBaseModel
from models.CNN import CNNet
from models.EncoderDecoder import Encoder_Decoder
from models.UNet import UNet
from models.UNet_enhance import UNet as UNet2
from models.ModelFactory import ModelFactory

class CNNHyperparamManager(object):
    """
    Class used for Hyperparameter search
    """

    # defining hyperparameters default values
    def_depths = [4]
    def_types = ['Large']



    def __init__(self, train_set, test_set, val_set, model_name: str, 
                batch_sizes : list, lr_list: list, optimizers : list, epochs : list,
                depths = def_depths, types = def_types
                 ):
        """
        Args:
            train_set : the training set
            test_set : the test set
            val_set : the validation set
            model_name : the name of the model
            batch_sizes : list of batch sizes
            lr : list of learning rates
            optimizers : list of optimizer factories
        """
        self.model_name = model_name
        self.batch_sizes = batch_sizes
        self.lr_list = lr_list
        self.optimizers = optimizers
        self.train_set = train_set
        self.test_set = test_set
        self.val_set = val_set
        self.num_epochs = epochs
        self.depths = depths
        self.types = types

        self.best_model = None
        self.best_batch_size = 0
        self.best_lr = 0
        self.best_opt_factory = None
        self.best_opt = None

    def tune(self):
        self.best_val_accuracy = 0
        self.best_val_accuracy_list = []

        num_classes = self.train_set.num_classes
        num_modalities = self.train_set.num_modalities


        #for model_name in self.model_name:

        # Get the factory of the current model
        model_factory = ModelFactory.getFactory(self.model_name)
        
        #if the model doesn't use depth parameter, we set the list to default for this parameter
        #  in order not to loop through un useless loop.
        if(not model_factory.has_option()):
            self.types = CNNHyperparamManager.def_types
        for type in self.types:
            if(not model_factory.has_depth()):
                self.depths = CNNHyperparamManager.def_depths
            for depth in self.depths: 
                for batch_size in self.batch_sizes:  # the loop is not directly on self.batch_sizes, because it can be changed during the execution
                    for lr in self.lr_list:
                        for opt in self.optimizers:
                            if opt == 'SGD':
                                optimizer_factory = optimizer_setup(optim.SGD, lr=lr, momentum=0.9)
                            elif opt == 'Adam':
                                optimizer_factory = optimizer_setup(optim.Adam, lr=lr)

                            self.test_hyperparams(model_factory, num_classes,num_modalities, batch_size, lr, optimizer_factory, opt, depth, type)
                        
                        
                        

    def test_hyperparams(self, model_factory, num_classes, num_modalities, batch_size, lr, optimizer_factory, opt, depth, type ):
        '''
        trains the model, calculate score and test if the loss is better than current best loss
        '''
        model = model_factory.getModel(num_classes=num_classes, in_channels=num_modalities, depth = depth, type = type )
        
        parameter_searcher = CNNTrainTestManager(model=model,
                                trainset=self.train_set,
                                testset=self.test_set,
                                batch_size=batch_size,
                                loss_fn=nn.CrossEntropyLoss(),
                                optimizer_factory=optimizer_factory,
                                validation=self.val_set,
                                learning_rate=lr,
                                use_cuda=True,
                                exp_name="",
                                verbose=False)

        start_epoch=0
        metric_values = None
        print("Training {} for {} epochs".format(
            model.__class__.__name__, self.num_epochs - start_epoch))

        parameter_searcher.train(self.num_epochs, start_epoch=start_epoch, metric_values=metric_values)

        val_accuracy = parameter_searcher.metric_values['val_acc'][-1]

        print("model: ",model.__class__.__name__,"type : ",type, "depth : ",depth," batch_size : ", batch_size, "lr: ", lr, "optimiser: ",opt," ===>   ",val_accuracy )

        if(val_accuracy > self.best_val_accuracy):
            print("Best val accuracy: ",val_accuracy)
            self.best_val_accuracy_list.append(val_accuracy)
            self.best_val_accuracy = val_accuracy
            self.best_model = model
            self.best_batch_size = batch_size
            self.best_lr = lr
            self.best_opt_factory = optimizer_factory
            self.best_opt = opt
