# Component Based Similar Image Search

Most of the reverse image searching is based on the image search as a whole. But it does not always catch the information regarding how inter-related two images are, based on their contents. A picture having a number of birds is far more related to a picture having a few birds and an animal than a third picture with all cars. So here a different approach has been used to capture this relation by matching the component of the images. A given composite image (with multiple objects) is searched among a set of other composite images and ordered based on how closely related it is with the images of the set. The top-most image in the ordering indicates the closest image to the given image. For component detection, selective search with fast non-maximal suppression has been used with ZCA normalization. The Convolutional neural network (CNN) have been used for the identification of the components. This can be used to find similarity among images which is difficult to find in conventional methods.

## High Level project Component :
1. Object Detection
2. Object Recognition


## Code Walkthrough
```
A_regionSelectionDriver.py      : Driver Module of Object Detection 
A_A_selectiveSearchFilter.py    : Applies selective search to extract regions from a given image
A_B_fastnms.py                  : Faster version of Non Maximal Suppression [Used Malisiewicz et al. version]
B_regionClasifierDriver_*.py    : Each of these is the driver for the Object Recognition. 
                                  Classifier having 2 Convolutional Layers[ReLU], 1 Fully-connected Layer[ReLU] and 1 Softmax Layer
B_A_Classifier.py               : Classifier Class (Used Michael Nielsen's verison(http://neuralnetworksanddeeplearning.com/)'s verison)
B_B_Evaluator.py                : Evaluates the classifier. Functionalities include checking for matches using overlap between ground truth and predicted area, finding accuracy, finding n top predictions
```

## Prerequisite
### Technical
```
python
**cuda** sudo apt-get install nvidia-cuda-toolkit
**opencv** sudo apt-get install python-opencv
**theano** http://deeplearning.net/software/theano/install.html
**selective search** pip install selectivesearch
CIFAR10 dataset
```

### Detection part
```
multi*.jpg      : 10 composite image used for the sample code
zcaParams.pkl   : Params from training data need to ZCA preprocess the test data
```

### Recognition part 
*[Can be run with added pretrained params]*
```
6060_1000_200_0.01_10.0.pkl : Pre-trained parameters received on training on CIFAR10 with mentioned values of hyperparameters
test.pkl                    : Represents the proposed preprocessed regions (test data)
```
