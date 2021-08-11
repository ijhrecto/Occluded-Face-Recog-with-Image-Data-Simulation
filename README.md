# Occluded-Face-Recog-with-Image-Data-Simulation
## Aim of Study
The aim of this study is to investigate current problems and study the accuracy of facial recognition on occluded facial image datasets. The scope of this study is the usage of artificial neural networks, and occlusions on the lower face area manifested by the usage of face masks for COVID-19. The researchers’ goal is to contribute by mitigating problems, such as usage of masks, for technologies using facial recognition on controlled environments like biometrics and authentication systems for workplaces and schools.

## Process
### Dataset
#### Facial Images - Georgia Tech Face
- http://www.anefian.com/research/face_reco.htm

#### Mask images
- included in the downloads
- functional design : N95 mask and medical masks
- abstract design : random shapes and patterns
- mouth design : mouths images or arts printed as a design

### Data Preprocessing
- The raw image data contains a person with a background, so we cropped and aligned the face.
- PIL and Dlib was used to process the raw images
- We created our own simulated occluded face images (occluded facial areas with black) for one our training experiments.
- We also simulated and superimpose a face mask on the raw facial images to create our own masked occluded facial image dataset

### Experiments
- We used a 22 layer neural network architecture to train our model. 
- We then classified the face using Support Vector Machine.
- There are 6 scenarios done to train and test different facial occlusion and facial mask.




