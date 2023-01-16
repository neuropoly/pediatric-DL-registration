# pediatric-DL-registration

# DLRegistrationFramework

The objective of this project is to use deep learning (DL) techniques for registering
pediatric brain MRI scans and allowing quicker processing time. 
DeepReg version 0.1.0 [1] is used to implement the unsupervised learning-based registration task. It is a tensorflow based implementation DL toolbox with unsupervised and weakly-supervised algorithms. The U-Net architecture was used and the output is a dense displacement field (DDF). One can easily train different networks using configuration files. The [config files](https://github.com/neuropoly/pediatric-DL-registration/tree/main/config_files) used in this work are available in this repository. DeepReg's GitHub repository [2] is available for further consultations with scripts coded in Python. 

# Requirements

DeepReg github repository from march 1st 2021 (version 0.1.0) and all its dependencies have to be 
installed in order to train the DL model with the yaml files. Refer to DeepReg
documentation for specifics around the variables in the configuration files. 
The [Calgary Preschool dataset](https://osf.io/axz5r/) is employed with its T1-weighted MRI images.
A total of 64 subjects is utilized to have at least two time-point images per
subject. The selected 64 subjects are presented in PatientDict.txt with the first 
column being the subject and the second all image scanIDs. <br />
Images were first N4 bias field corrected as well as rescaled to 1.5 mm isotropic resolution using FLIRT version 6.0 (-applyisoxfm option).
Template-based registration (with every image registered in a pediatric template space; [NIST 4.5-8.5 y.o. pediatric atlas](http://nist.mni.mcgill.ca/pediatric-atlases-4-5-18-5y/)) was used to obtain white matter (WM), gray matter (GM) and cerebrospinal fluid (CSF) segmentations for each subject for validation puposes. 

# Procedure

* Pair-based registration (with registration done on all possible pairs (431 pairs))

The [scripts folder](https://github.com/neuropoly/pediatric-DL-registration/tree/main/scripts) contains multiple functions and bash scripts for: <br /> 
* Training all intra-subject pairs
* Evaluating the registration learning-based approaches on segmentations after warping
* Jacobian determinant calculations
* Time calculations
ANTs commands used to pre-register the images are available in.... Finally, some basic functions for plotting results are also available in ....

# References

[1]DeepReg. Image Registration with Deep Learning. 2021. url: https://deepreg.readthedocs.io/en/latest/tutorial/registration.html. <br />
[2]DeepReg. Medical image registration using deep learning. 2021. url: https://github.com/DeepRegNet/DeepReg.

## Citing this work
If some of these implementations helped you, please don't hesitate to cite the followings:
A. Dimitrijevic, V. Noblet, and B. De Leener, “Deep Learning-Based Longitudinal Intra-subject Registration of Pediatric Brain MR Images,” in Biomedical Image Registration, 2022, pp. 206–210.
-...

