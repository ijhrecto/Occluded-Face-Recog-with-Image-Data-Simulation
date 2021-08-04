### the files to the research paper, to create a method for facial recognition on masked faces 

# Mask Occluded Face Recognition on a Controlled Environment using Convolutional Neural Networks

### Abstract
Facial recognition is a widely used technology in recent years. Face masks are a common and an additional partial occlusion on faces. Due to the COVID-19 virus, there is an increase of face mask usage even inside the workplace. It contributes to the factors degrading the efficiency of the recognition system. This study focuses on an occlusion factor manifested by face masks with a varying design and category. The researchers proposed an approach by using diverse occluded face dataset for the training data. We processed both the face image and mask images using dlib to create an occluded training and testing image dataset.  For facial features, extraction was done by embedding the faces using FaceNet architecture. The faces were classified using Support Vector Machine. We experimented with different scenarios by using different training sets and testing sets. It achieved a performance of recognizing occluded lower face images with an average accuracy rate of 98.93% on a controlled environment.

### Aim of Study
The aim of this study is to investigate current problems and study the accuracy of facial recognition on occluded facial image datasets. The scope of this study is the usage of artificial neural networks, and occlusions on the lower face area manifested by the usage of face masks for COVID-19. The researchers’ goal is to contribute by mitigating problems, such as usage of masks, for technologies using facial recognition on controlled environments like biometrics and authentication systems for workplaces and schools.

### Data Simulation
The raw image data contains a person with a background, so we cropped and aligned the face.
We created our own simulated occluded face images (occluded facial areas with black) for one our training experiments.
We also simulated and superimpose a face mask on the raw facial images to create our own masked occluded facial image dataset

### Experiments
For facial extractioni we used a 22 layer neural network architecture to train our model. 
We then classified the face using Support Vector Machine.
There are 6 scenarios done to train and test different facial occlusion and facial mask.

### Dataset
RESOURCE DATASET
Georgia Tech Face
- http://www.anefian.com/research/face_reco.htm

Mask images
- included in the downloads
- functional design : N95 mask and medical masks
- abstract design : random shapes and patterns
- mouth design : mouths images or arts printed as a design


