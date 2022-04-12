# -*- coding:utf-8 -*-

"""
Remarque importante sur la manipulation des images avec scikit-image et cv2:
Pour passer d'une image importer avec skimage (scikit-image) à une image importer avec CV2
il faut faire: image = img_as_float(image) 
et inversement: image = img_as_ubyte(image)
le code part d'un tableau numpy qui est créer avec cv2 donc ne pas oublier d'effectuer la transformation si vous
importez vos images avec scikit-image.
"""

from copy import copy
import numpy as np
import imutils
import random
import cv2
from skimage.util import random_noise
from skimage import transform,img_as_ubyte,img_as_float
from PIL import Image,ImageEnhance


class Transformation():

    """               
    Contient toutes les fonctions utiles pour augmenter la taille de nos images
    La première partie contient les fonctions qui réalisent une seul opération sur les images, la suite contient la 
    fonction qui permet de générer des images à partir d'une image d'origine en fonction de nombreux critères. Elle
    permet de générer autant d'image toutes différentes et proches de l'image originale pour augmenter la taille de 
    différent type de data.
    """

    def __init__(self,number_of_generation=10,transformation_list=[],angle=(-15,15),start_pose=2,translation_range=(-50,50),shear_range=(-0.1,0.1),contrast_range=(0.8,1.3),brightness_range=(0.9,1.2),sharpness_range=(0.85,1.1), color_range=(0.85,1.2)):
        
        """
        Les arguments sont à renseigner pour utiliser la fonction generation aléatoire sinon il n'est pas utile de les renseigner,
        vous pouvez tout à fait utiliser les fonctions indépendamment en se référant à la note ci-dessous
        Args:
            -number_of_generation = number of image to generate with our starting image
            -transformation_list = list des fonctions de la classe Augmentation à utiliser lors de la génération aléatoire
            -angle = tuple contenant l'angle minimal et l'angle maximal de rotation de notre image de départ
            -start_pose=position de départ pour l'image sur laquelle on applique un angle 1=0°; 2=0° et 180°; 3=0°,90°,180°,270°
            -translation_range = tuple contenant le nombre pixel minimal et maximal de décalage horizontal et vertical de notre image
            -shear_range = tuple contenant le rapport minimal et maximal de cisaillement de notre image
            -contrast_range = idem mais pour le contraste
            -brightness_range = idem pour la luminosité
            -sharpness_range = idem pour l'accentuation
            -color_rante = idem pour la saturation des couleurs
        """

        self.number_of_generation=number_of_generation
        if(transformation_list==[]):
            self.transformation_list = [self.rotate,self.translate,self.flip,self.shear,self.noise2,self.contrast,self.brightness,self.sharpness,self.color]
        else:
            self.transformation_list = transformation_list
        self.angle = angle
        self.start_pos=start_pose
        self.translation_range = translation_range
        self.shear_range = shear_range 
        self.contrast_range = contrast_range
        self.brightness_range = brightness_range 
        self.sharpness_range = sharpness_range
        self.color_range = color_range

    ###########################################################
    #             TRANSFORMATIONS GEOMETRIQUES                #
    ###########################################################

    """
    Note : pour utiliser les fonctions indépendemment les une des autres, mettre mode='exact' et renseigner les arguments
    propre à la fonction, sinon laisser en mode aléatoire et modifier l'init pour utiliser une génération aléatoire.
    
    Pour modifier la prédiction de la même façon que la fonction il suffit de la passer dans le paramêtre prediction.
    """

    def rotate(self,image,mode="random",degree=0,prediction=np.zeros(shape=1))->np.ndarray:
        ''' pour faire effectuer une rotation à une image, mode = 'exact' degré = (int) entre 0 et 360  '''
        
        #gestion des modes
        if(mode=="random"):
            angle=random.randint(self.angle[0],self.angle[1])
            if(self.start_pos==2):
                r2=random.randint(0,1)
                angle+=180*r2
            if(self.start_pos==4):
                r2=random.randint(0,3)
                angle+=90*r2
        elif(mode=="exact"):
            angle=degree
        else:
            angle=50
            print("error wrong parameters in fonction rotate, mode should be either random or exact")
        
        #corps de la fonction
        rotate_image = imutils.rotate_bound(image, angle)
        image_finale=cv2.resize(rotate_image,(900,900))
    
        #si la prédiction est passé en argument 
        if(prediction.shape!=np.zeros(shape=1).shape):
            rotate_prediction = imutils.rotate_bound(prediction, angle)
            prediction_finale=cv2.resize(rotate_prediction,(900,900))
            return image_finale,prediction_finale

        return image_finale

    def translate(self,image,mode="random",horizontal_translation=0,vertical_translation=0,prediction=np.zeros(shape=1))->np.ndarray:
        ''' horizontal_translation = décalage horizontale en pixel, vertical_translation = décalage vertical en pixel'''

        #gestion des modes
        if(mode=="random"):
            horizontal_translation=random.randint(self.translation_range[0],self.translation_range[1])
            vertical_translation=random.randint(self.translation_range[0],self.translation_range[1])
        elif(mode=="exact"):
            horizontal_translation=horizontal_translation
            vertical_translation=vertical_translation
        else:
            horizontal_translation=0
            vertical_translation=0
            print("error wrong parameters in fonction translate, mode should be either random or exact")

        #corps de la fonction
        translate_image = imutils.translate(image, horizontal_translation, vertical_translation)
        image_finale=cv2.resize(translate_image,(900,900))

        #si la prédiction est passé en argument 
        if(prediction.shape!=np.zeros(shape=1).shape):
            translate_prediction = imutils.translate(prediction, horizontal_translation, vertical_translation)
            prediction_finale=cv2.resize(translate_prediction,(900,900))
            return image_finale,prediction_finale

        return image_finale

    def flip(self,image,mode="random",option=1,prediction=np.zeros(shape=1))->np.ndarray:
        ''' option = 1 pour renverser horizontalement, 0 pour verticalement '''
        #gestion des modes
        if(mode=="random"):
            option=random.randint(0,1)
            if(option==2):
                if(prediction.shape!=np.zeros(shape=1).shape):
                    return image,prediction
                return image
        elif(mode=="exact"):
            option=option
        else:
            option=1
            print("error wrong parameters in fonction flip, mode should be either random or exact")

        #corps de la fonction
        flip_image = cv2.flip(image, option)

        #si la prédiction est passé en argument

        if(prediction.shape!=np.zeros(shape=1).shape):
            flip_prediction = cv2.flip(prediction, option)
            return flip_image,flip_prediction

        return flip_image

    def shear(self,image,mode="random",factor=0,prediction=np.zeros(shape=1)):
        ''' choisir facteur entre -1 et 1, rester proche de 0 pour de bonnes performances (>0 incline vers la droite, <0 vers la gauche)'''

        #gestion des modes
        if(mode=="random"):
            factor=random.randint(int(self.shear_range[0]*100),int(self.shear_range[1]*100))/100
        elif(mode=="exact"):
            factor=factor
        else:
            factor=0
            print("error wrong parameters in fonction shear, mode should be either random or exact")
        
        #corps de la fonction
        # comme skimage et cv2 ont des formats d'encodage différents:
        image =img_as_float(image)
        # Create Afine transform
        afine_tf = transform.AffineTransform(shear=factor)
        # Apply transform to image data
        modified = transform.warp(image, inverse_map=afine_tf)
        # et on revient à notre CV2:
        modified=img_as_ubyte(modified)

        #si la prédiction est passé en argument
        if(prediction.shape!=np.zeros(shape=1).shape):
            # comme skimage et cv2 ont des formats d'encodage différents:
            prediction =img_as_float(prediction)
            # Apply transform to image data
            modified_pred = transform.warp(prediction, inverse_map=afine_tf)
            # et on revient à notre CV2:
            modified_pred=img_as_ubyte(modified_pred)
            return modified,modified_pred

        return modified


    ###########################################################
    #               TRANSFORMATIONS DIVERSES                  #
    ###########################################################

    def noise2(self,image,mode="random",type="gaussian",prediction=np.zeros(shape=1)):

        ''' choisir parmis les options: pepper, gaussian, poisson, s&p, speckle'''
        # si bruit trop important utiliser la fonction noise1

        #gestion des modes
        if(mode=="random"):
            l=["pepper","gaussian","poisson","s&p","speckle"]
            r=random.randint(0,4)
            type=l[r]
        elif(mode=="exact"):
            type=type
        else:
            type="gaussian"
            print("error wrong parameters in fonction noise2, mode should be either random or exact")

        #corps de la fonction
        # comme skimage et cv2 ont des formats d'encodage différents:
        image =img_as_float(image)
        noise_image = random_noise(image, mode=type)
        # et on revient à notre CV2:
        noise_image=img_as_ubyte(noise_image)
        if(prediction.shape!=np.zeros(shape=1).shape):
            return noise_image,prediction
        return noise_image

    def noise(image:np.ndarray,noise_typ):
        ''' version developpée de noise 2 pour plus de  contrôle, pourrait être utile donc à conserver même si moins performante que noise()'''
        
        if noise_typ == "gauss":
            row,col,ch= image.shape
            mean = 0
            var = 5
            sigma = var**0.5
            gauss = np.random.normal(mean,sigma,(row,col,ch))
            gauss = gauss.reshape(row,col,ch)
            noisy = image + gauss
            return noisy
        elif noise_typ == "salt":
            row,col,ch = image.shape
            s_vs_p = 0.5
            amount = 0.01
            out = np.copy(image)
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                    for i in image.shape]
            out[coords] = 1

            # Pepper mode
            num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                    for i in image.shape]
            out[coords] = 0
            return out
        elif noise_typ == "poisson":
            vals = len(np.unique(image))
            vals = 3 ** np.ceil(np.log2(vals))
            noisy = np.random.poisson(image * vals) / float(vals)
            return noisy
        elif noise_typ =="speckle":
            row,col,ch = image.shape
            gauss = np.random.randn(row,col,ch)
            gauss = gauss.reshape(row,col,ch)        
            noisy = image + image * gauss
            return noisy

    def contrast(self,image,mode="random",factor=1,prediction=np.zeros(shape=1)):
        ''' pour changer le contrast d'une image, facteur entre 0 et 4: facteur < 1 = reduit les contrastes, >1 = augmente les contrastes '''
        
        #gestion des modes
        if(mode=="random"):
            factor=random.randint(int(self.contrast_range[0]*100),int(self.contrast_range[1]*100))/100
        elif(mode=="exact"):
            factor=factor
        else:
            factor=0
            print("error wrong parameters in fonction contrast, mode should be either random or exact")

        #corps de la fonction
        pil_image=Image.fromarray(image)
        img_contr_obj=ImageEnhance.Contrast(pil_image)
        new_pil_img=img_contr_obj.enhance(factor)
        new_img=np.array(new_pil_img)

        #si la prédiction est passé en argument 
        if(prediction.shape!=np.zeros(shape=1).shape):
            '''
            pil_image=Image.fromarray(prediction)
            img_contr_obj=ImageEnhance.Contrast(pil_image)
            new_pil_img=img_contr_obj.enhance(factor)
            new_img_pred=np.array(new_pil_img)
            return new_img,new_img_pred
            '''
            return new_img,prediction

        return new_img

    def brightness(self,image,mode="random",factor=1,prediction=np.zeros(shape=1)):
        ''' pour changer la luminosité d'une image, facteur entre 0 et 4: facteur < 1 = reduit la luminosité, >1 = augmente la luminosité '''
        
        #gestion des modes 
        if(mode=="random"):
            factor=random.randint(int(self.brightness_range[0]*100),int(self.brightness_range[1]*100))/100
        elif(mode=="exact"):
            factor=factor
        else:
            factor=0
            print("error wrong parameters in fonction brightness, mode should be either random or exact")
        
        #corps de la fonction
        pil_image=Image.fromarray(image)
        img_bright_obj=ImageEnhance.Brightness(pil_image)
        new_pil_img=img_bright_obj.enhance(factor)
        new_img=np.array(new_pil_img)

        #si la prédiction est passé en argument
        if(prediction.shape!=np.zeros(shape=1).shape):
            '''
            pil_image=Image.fromarray(prediction)
            img_contr_obj=ImageEnhance.Brightness(pil_image)
            new_pil_img=img_contr_obj.enhance(factor)
            new_img_pred=np.array(new_pil_img)
            return new_img,new_img_pred
            '''
            return new_img,prediction
        return new_img

    def sharpness(self,image,mode="random",factor=1,prediction=np.zeros(shape=1)):
        ''' pour accentuer une image, le facteur doit être entre 0 et 4: facteur < 1 = reduit l'accentuation, >1 = augmente l'accentuation '''
        
        #gestion des modes
        if(mode=="random"):
            factor=random.randint(int(self.sharpness_range[0]*100),int(self.sharpness_range[1]*100))/100
        elif(mode=="exact"):
            factor=factor
        else:
            factor=0
            print("error wrong parameters in fonction sharpness, mode should be either random or exact")

        #corps de la fonction
        pil_image=Image.fromarray(image)
        img_sharp_obj=ImageEnhance.Sharpness(pil_image)
        new_pil_img=img_sharp_obj.enhance(factor)
        new_img=np.array(new_pil_img)

        #si la prédiction est passé en argument
        if(prediction.shape!=np.zeros(shape=1).shape):
            '''
            pil_image=Image.fromarray(prediction)
            img_contr_obj=ImageEnhance.Sharpness(pil_image)
            new_pil_img=img_contr_obj.enhance(factor)
            new_img_pred=np.array(new_pil_img)
            return new_img,new_img_pred
            '''
            return new_img,prediction
        return new_img

    def color(self,image,mode="random",factor=1,prediction=np.zeros(shape=1)):
        ''' pour saturer une image au niveau des couleurs facteur entre 0 et 4 facteur < 1 = reduction des couleurs, >1 = saturation des couleurs '''
        
        #gestion des modes
        if(mode=="random"):
            factor=random.randint(int(self.color_range[0]*100),int(self.color_range[1]*100))/100
        elif(mode=="exact"):
            factor=factor
        else:
            factor=0
            print("error wrong parameters in fonction color, mode should be either random or exact")

        #corps de la fonction
        pil_image=Image.fromarray(image)
        img_color_obj=ImageEnhance.Color(pil_image)
        new_pil_img=img_color_obj.enhance(factor)
        new_img=np.array(new_pil_img)

        #si la prédiction est passé en argument
        if(prediction.shape!=np.zeros(shape=1).shape):
            '''
            pil_image=Image.fromarray(prediction)
            img_contr_obj=ImageEnhance.Color(pil_image)
            new_pil_img=img_contr_obj.enhance(factor)
            new_img_pred=np.array(new_pil_img)
            return new_img,new_img_pred
            '''
            return new_img,prediction
        return new_img

    ###########################################################
    #                    FONCTION FINALE                      #
    ###########################################################

    def apply_transformations(self,image,prediction=np.zeros(shape=1)):
        if(prediction.shape!=np.zeros(shape=1).shape):
            generate_list=[(image,prediction)]
        else:
            generate_list=[image]

        for k in range(self.number_of_generation):
            working_image=np.copy(image)
            working_prediction=np.copy(prediction)

            for fun in self.transformation_list:
                if(working_prediction.shape!=np.zeros(shape=1).shape):
                    working_image,working_prediction=fun(working_image,prediction=working_prediction)
                else:
                    working_image=fun(working_image)

            if(working_prediction.shape!=np.zeros(shape=1).shape):
                generate_list.append((working_image,working_prediction))
            else:
                generate_list.append(working_image)

        return generate_list

