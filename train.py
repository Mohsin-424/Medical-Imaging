#!/usr/bin/env python
# -*- coding:utf-8 -*-



try:
  from google.colab import files
  IN_COLAB = True
except:
  IN_COLAB = False

import argparse
import warnings

import torch
import torch.nn as nn
import torch.optim as optim

from os import mkdir, listdir
from os.path import join, exists

from CNNTrainTestManager import CNNTrainTestManager, optimizer_setup
from models.ModelFactory import ModelFactory
from CNNHyperparameterSearch import CNNHyperparamManager
from HDF5Dataset import HDF5Dataset

from transforms import identity

# Define hyperparameters with help section, default value and choices if needed
hyperparams = {
    'lr' : ['Learning rate', 0.01, None],
    'optimizer' : ["The optimizer to use for training the model" , "Adam", ["Adam", "SGD"]],
    'batch_size' : ['The size of the training batch', 20, None],
    'model_depth' : ['depth of the model, usable for UNet or Encoder-Decoder', 4, range(1,7)],
    'model_version' : ['version of the model, usable for OurNet or MSCADC', 'Large', ['Large', 'Medium', 'Small']]
}


def argument_parser():
    """
        A parser to allow user to easily experiment different models along with
        datasets and differents parameters
    """
    parser = argparse.ArgumentParser(usage='\n python3 train.py [model] [dataset] [hyper_parameters]'
                                           '\n python3 train.py --model UNet [hyper_parameters]'
                                           '\n python3 train.py --model UNet --predict',
                                     description="This program allows to train different models of classification on"
                                                 " different datasets. Be aware that when using UNet model there is no"
                                                 " need to provide a dataset since UNet model only train "
                                                 "on acdc dataset.")
    parser.add_argument("exp_name", type=str,
                        help="Name of experiment")
    parser.add_argument('--model', type=str, default="CNNet",
                        choices=ModelFactory.model_dict.keys())
    parser.add_argument('--dataset_file', type=str,
                        help="Location of the hdf5 file")
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='The number of epochs')
    parser.add_argument('--validation', type=float, default=0.1,
                        help='Percentage of training data to use for validation')
    parser.add_argument('--data_aug', action='store_true',
                        help="Data augmentation")
    parser.add_argument('--predict', action='store_true',
                        help="Load weights to predict the mask of a randomly selected image from the test set")
    parser.add_argument('--tune', action='store_true',
                        help="Use hyperparameters tuning")
    parser.add_argument('--kfold', type=int, default=1,
                        help='The number of validation sets for cross validation')
    parser.add_argument('--resume', action='store_true',
                        help="Resume training on last epoch done")
    parser.add_argument('--download', action='store_true',
                        help="On google colab download the images in the experiment folder")

    # we add an entry for every hyperparameters, they all can have 0,1 or more values
    for key, value in hyperparams.items():
        parser.add_argument(f'--{key}', type=type(value[1]), default=[value[1]],
                        help=value[0], choices=value[2], nargs='*')
    
    return parser.parse_args()


if __name__ == "__main__":

    # get general values
    args = argument_parser()
    data_augment = args.data_aug    
    num_epochs = args.num_epochs
    val_set = args.validation
    resume = args.resume

    # check if experiment directory all-ready exists
    if exists(args.exp_name):
            warnings.warn(f"The directory {args.exp_name} already exists, some files may be overwritten")
    else:
        mkdir(args.exp_name)

    # set hdf5 path according your hdf5 file location
    hdf5_file = args.dataset_file

    # Transform is used to normalize data and more
    data_augment_transform = [
        identity
    ]
    if data_augment:
        print('Using data augmentation')
        transforms = data_augment_transform
    else:
        print("Not using data augmentation")
        transforms = []

    train_set = HDF5Dataset(
        'train', hdf5_file, transforms=transforms)
    test_set = HDF5Dataset(
        'test', hdf5_file, transforms=transforms)
    num_classes = train_set.num_classes
    num_modalities = train_set.num_modalities

    best_val_loss = 0

    #model_class = model_dict[args.model]
    model_name = args.model

    ### hyperparameters
    # if tune is selected, get the best hyperparameters
    if args.tune:
        list_batch_sizes = args.batch_size
        list_lr = args.lr
        list_optimizers = args.optimizer
        list_depths = args.model_depth
        model_version=args.model_version

        hyperparamManager = CNNHyperparamManager(train_set, test_set, val_set, model_name = model_name,
                                                 batch_sizes = list_batch_sizes, lr_list = list_lr, 
                                                 optimizers = list_optimizers, epochs = num_epochs,
                                                 depths = list_depths, types = model_version)
        hyperparamManager.tune()

        best_val_accuracy_list = hyperparamManager.best_val_accuracy_list
        print("list of the best accuracies: ", best_val_accuracy_list)

        best_val_accuracy = hyperparamManager.best_val_accuracy


        model = hyperparamManager.best_model
        batch_size = hyperparamManager.best_batch_size
        learning_rate = hyperparamManager.best_lr
        optimizer_factory = hyperparamManager.best_opt_factory
        opt = hyperparamManager.best_opt

        print("\\\\\            The best model is: ",model.__class__.__name__,"          ///// \n \
                        with HYPERPARAMETERS : \n \
        batch_size : ", batch_size, "lr: ", learning_rate, "optimiser: ",opt,"\n \
                        with an accuracy of = ",best_val_accuracy,)

    else:
        if len(args.lr) != 1 or len(args.batch_size) != 1 or len(args.optimizer) != 1:
            raise ValueError("Parameter have several values without tuning")

        batch_size = args.batch_size[0]
        learning_rate = args.lr[0]
        model_factory = ModelFactory.getFactory(model_name)
        model = model_factory.getModel(num_classes=num_classes, in_channels=num_modalities, option=args.model_version[0], depth=args.model_depth[0])

        if args.optimizer[0] == 'SGD':
            optimizer_factory = optimizer_setup(
                torch.optim.SGD, lr=learning_rate, momentum=0.9)
        elif args.optimizer[0] == 'Adam':
            optimizer_factory = optimizer_setup(optim.Adam, lr=learning_rate)

    model_trainer = CNNTrainTestManager(model=model,
                                        trainset=train_set,
                                        testset=test_set,
                                        batch_size=batch_size,
                                        loss_fn=nn.CrossEntropyLoss(),
                                        optimizer_factory=optimizer_factory,
                                        validation=val_set,
                                        learning_rate=learning_rate,
                                        use_cuda=True,
                                        exp_name=args.exp_name)
    if args.predict:
        model.load_weights(join(args.exp_name, model.__class__.__name__ + '.pt'))
        #model_trainer.evaluate_on_test_set()
        print("predicting the mask of a randomly selected image from test set")
        model_trainer.plot_image_mask_prediction(args.exp_name, 'prediction')
    else:
        start_epoch=0
        metric_values = None
        if resume :
            start_epoch, metric_values = model.load_checkpoint(join(args.exp_name, model.__class__.__name__ + '_temp.pt'))
            print("Resume model training from epoch", start_epoch)
        print("Training {} for {} epochs".format(
            model.__class__.__name__, args.num_epochs - start_epoch))
        model_trainer.train(args.num_epochs, start_epoch=start_epoch, metric_values=metric_values)
        model_trainer.evaluate_on_test_set()
        # save the model's weights for prediction (see help for more details)
        model.save(args.exp_name)
        model_trainer.plot_image_mask_prediction(args.exp_name, 'fig2')
        model_trainer.plot_metrics(args.exp_name)

    if args.download and IN_COLAB:
        dir = args.exp_name
        file_list = listdir(dir)
        for f in file_list:
            if f.endswith('.png'):
                files.download(join(dir, f))