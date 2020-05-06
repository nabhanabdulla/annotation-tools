# Annotation Tools

Data annotation is crucial for machine learning as our model is only as better as our data. 
But, annotating data is a tedious process. This repo provides GUI tools to improve the 
data annotation experience and reduce time consumed for the same

# Tools Available

## Pose Estimation

1. [Single/Dual Person Pose Estimation](https://github.com/nabhanabdulla/annotation-tools/tree/master/Single%20Person%20Pose%20Estimation)
* Provides a GUI tool made using TKinter for annotating atmost two people in an image
* Coordinates are saved by clicking on the loaded image in the order of the keypoints required

2. [Multi Person Pose Estimation](https://github.com/nabhanabdulla/annotation-tools/tree/master/OpenPoseKit)
* Builds on [DeepPoseKit](http://deepposekit.org) tool to add functionality to annotate multiple
persons in an image
* Added support for using output of pre-trained [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose/)
model to avoid annotation from scratch
* Added features like resetting keypoints and deleting images for enhanced experience when
using the GUI application
