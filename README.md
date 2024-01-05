# pediatric-DL-registration

## Objectives
The objective of this project is to use deep learning (DL) techniques for registering
pediatric brain MRI scans and allowing quicker processing time. 
[DeepReg version 0.0.0](https://deepreg.readthedocs.io/en/develop/tutorial/registration.html) [1] is used to implement the unsupervised learning-based registration task. The only main modification is the removal of affine data augmentation in the automatic pre-processing steps available in their framework. It is a tensorflow based implementation DL toolbox with unsupervised and weakly-supervised algorithms. The U-Net architecture was used and the output is a dense displacement field (DDF). One can easily train different networks using configuration files. The [config files](https://github.com/neuropoly/pediatric-DL-registration/tree/main/config_files) used in this work are available in this repository. DeepReg's GitHub repository [2] is available for further consultations with scripts coded in Python. 

## Requirements

DeepReg github repository from march 1st 2021 (version 0.0.0) and all its dependencies have to be 
installed in order to train the DL model with the yaml files. Refer to DeepReg
documentation for specifics around the variables in the configuration files. To install all dependencies related to running DeepReg, run the following line of code to create a deepreg environment:

```
conda env create -f env_deepreg.yml
```

## Dataset
The [Calgary Preschool dataset](https://osf.io/axz5r/) is employed with its T1-weighted MRI images.
A total of 64 subjects is utilized to have at least two time-point images per
subject. The selected 64 subjects are presented in PatientDict.txt with the first 
column being the subject and the second all image scanIDs. <br />

## Preprocessing
Images were first N4 bias field corrected were inputted to [SynthSeg](https://surfer.nmr.mgh.harvard.edu/fswiki/SynthSeg) version 2.0 to obtain 18 brain regions of interest for validation purposes. Then, images were rescaled to 1.5 mm isotropic resolution using FLIRT version 6.0 (-applyisoxfm option).

## Procedure

The pair-based registration (with registration done on all possible pairs (434 pairs)) evaluated three types of initialization approaches after pre-processing steps:

* Non previously registered intra-subject pairs (NoReg)
* Rigidly registered via ANTs intra-subject pairs (RigidReg)
* Rigid and affine registered via ANTs intra-subject pairs (RigidAffineReg)

These three different inputs were used and compared SyN ANTs as shown in the figure below:
![](/images/fig-1.png "Scheme of all three initialization approaches used")

As for the full pipeline, it is visible below:
![](/images/fig-a3.png "Full pipeline")

The [scripts folder](https://github.com/neuropoly/pediatric-DL-registration/tree/main/scripts) contains multiple functions and bash scripts for: <br /> 
* Training all intra-subject pairs
* Evaluating the registration learning-based approaches on segmentations after warping
* Jacobian determinant calculations
* Time calculations
ANTs commands used to pre-register the images are available in [DataHandle.py](https://github.com/neuropoly/pediatric-DL-registration/blob/main/scripts/DataHandle.py) where the [ANTs version](https://github.com/ANTsX/ANTs/releases) used is 2.3.4.dev172-gc801b.

## Analyses

Graphs depicting Dice score results in relation to the age interval between pairs are included in the article for white matter, gray matter, and cerebrospinal spinal fluid. Animated graphs have been generated to facilitate a more detailed examination of Dice scores per age interval for each pair at a local level across all 18 segmented regions. These regions are calculated by averaging right and left hemispheres for every region, except for the brain-stem, 3rd ventricle, 4th ventricle, and CSF, which are considered as a whole in the initial 32 given labels. The provided graphs illustrate the results for [NoReg](https://neuropoly.github.io/pediatric-DL-registration/AgePlot_NoReg.html), [RigidReg](https://neuropoly.github.io/pediatric-DL-registration/AgePlot_RigidReg.html), and [RigidAffineReg](https://neuropoly.github.io/pediatric-DL-registration/AgePlot_RigidAffineReg.html), with ANTs SyN Reg in red, DL Reg in green, and the initial alignment in blue.

## References

[1]DeepReg. Image Registration with Deep Learning. 2021. url: https://deepreg.readthedocs.io/en/latest/tutorial/registration.html. <br />
[2]DeepReg. Medical image registration using deep learning. 2021. url: https://github.com/DeepRegNet/DeepReg.

## Citing this work
If some of these implementations helped you, please don't hesitate to cite the followings:
- A. Dimitrijevic, V. Noblet, and B. De Leener, “Deep Learning-Based Longitudinal Intra-subject Registration of Pediatric Brain MR Images,” in Biomedical Image Registration, 2022, pp. 206–210.
-...

