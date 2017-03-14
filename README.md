# House Number Recognition on the Street View House Number (SVHN) dataset

## Project Overview

Human brain is pretty good at gathering high-level information from images. However, these tasks are often challenging for machines. Computer vision is an interdisciplinary field seeks to automate tasks that the human visual system can do. One of these
interesting tasks is to recognize digits in natural scenes.

The Street View House Numbers (SVHN) Dataset [1] is a collection of images contain house numbers in natural scenes created by Google. For Google, being able to read house numbers from images means improving map quality since every house number picture is geotagged. In this project, I will use deep learning approach based on Goodfellow [2] to recognize house numbers in the SVHN Dataset.

## Problem Statement

This problem is a supervised classification task. Given an image which contains a house number, the model should output a sequence of digits as a prediction. This problem is similar but more difficult than classifying digits for the MNIST dataset [3], because of the
following two reasons:
1. The SVHN dataset contains images from natural scene, where perspective, lighting conditions, and other objects can cause distraction.
2. For a house number prediction to be considered as correct, all the digits in the house number should match the target label.

Generally speaking, house numbers range from 1 to 99999, and most of the house numbers in the SVHN dataset have 2-4 digits. Consider the size of our dataset and the distribution of house numbers’ lengths, it would be impractical (way too many) if we define 99999 classes for this problem.

In this project, I will be assuming that house numbers are 1-5 digits long and defining 11 classes for each digit. Class [0-9] represents digit value [0-9], and class 10 represents N/A. The final model will take an image which contains a house number with 1-5 digits long, and output a sequence of 1-5 digits as a prediction. I will be building this model using deep convolutional neural networks based on Goodfellow’s approach [2].

## Reference
1. The Street View House Numbers (SVHN) Dataset: http://ufldl.stanford.edu/housenumbers/
2. Ian J. Goodfellow, Yaroslav Bulatov, Julian Ibarz, Sacha Arnoud, Vinay Shet (2014). Multi-digitNumber Recognition from Street View Imagery using Deep Convolutional Neural Networks. https://arxiv.org/pdf/1312.6082.pdf

**Note**

Version of Python:  Python 2.7

Libraries: tensorflow, numpy, scipy, matplotlib, cPickle


Preprocessing: 			1_SVHN_Preprocessing.ipynb
						
						
						Generates: 	SVHN_metadata.pickle and SVHN_data.pickle


CNN Model Tuning: 		2_SVHN_CNN_tuning.ipynb

						Generates: CNN_trained_initialModel.ckpt and CNN_trained_refinedModel.ckpt


CNN Final Model: 		3_SVHN_CNN_final.ipynb

    					Generates: CNN_trained_finalModel.ckpt

Predictions: 			4_SVHN_Predict.ipynb


Model visualization: 	utils.py 
