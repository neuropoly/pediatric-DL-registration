# -*- coding: utf-8 -*-
# This code for N4 bias field correction was adapted from https://github.com/dpaniukov/CBF_early_childhood/blob/master/preproc/reg_t12mni_N4corr.py

import os, sys  # system functions
import nipype.interfaces.io as nio  # Data i/o
import nipype.interfaces.fsl as fsl  # fsl
import nipype.pipeline.engine as pe  # pypeline engine
import nipype.interfaces.utility as util  # utility
import nipype.interfaces.ants as ants
from nipype.interfaces.c3 import C3dAffineTool
from nipype.interfaces.ants.segmentation import BrainExtraction

import multiprocessing, time
from multiprocessing import Pool
nprocs=multiprocessing.cpu_count()

start_time = time.time()


fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

"""
Project info
"""
# which subjects to run
subject_id = str(sys.argv[1])
scan_id = str(sys.argv[2])


main_dir = "/home/andjela/"
project_dir = main_dir+"Documents/Preprocess/"
work_dir = main_dir+"Documents/work_dir/"

if not os.path.exists(work_dir):
    os.makedirs(work_dir)

template_brain = os.path.join(project_dir,'MNI','nihpd_asym_04.5-08.5_t1w.nii')
anat_image = project_dir+"OSFData/"+subject_id+"/"+scan_id+"/"+scan_id+ ".nii.gz"

# """
# Create workflow
# """

wf = pe.Workflow(name='wf')
wf.base_dir = os.path.join(work_dir, "reg_n4_wdir", subject_id, scan_id)
wf.config = {"execution": {"crashdump_dir": os.path.join(work_dir, 'reg_n4_crashdumps', subject_id, scan_id)}}


# """
# N4 bias correction
# """

n4 = pe.Node(ants.N4BiasFieldCorrection(), name='n4')
n4.inputs.dimension = 3
n4.inputs.input_image = anat_image
n4.inputs.bspline_fitting_distance = 300
n4.inputs.shrink_factor = 3
n4.inputs.n_iterations = [50,50,50,50]
n4.inputs.num_threads = nprocs


# """
# Register T1 to MNI
# """
reg = pe.Node(ants.Registration(), name='antsRegister')
reg.inputs.output_transform_prefix = "anat2mni_"
reg.inputs.transforms = ['Rigid', 'Affine', 'SyN']
reg.inputs.transform_parameters = [(0.1,), (0.1,), (0.2, 3.0, 0.0)]
reg.inputs.number_of_iterations = [[10000,11110,11110]] * 2 + [[100, 100, 50]]
reg.inputs.dimension = 3
reg.inputs.write_composite_transform = True
reg.inputs.collapse_output_transforms = True
reg.inputs.initial_moving_transform_com = True
reg.inputs.metric = ['Mattes'] * 2 + [['Mattes', 'CC']]
reg.inputs.metric_weight = [1] * 2 + [[0.5, 0.5]]
reg.inputs.radius_or_number_of_bins = [32] * 2 + [[32, 4]]
reg.inputs.sampling_strategy = ['Regular'] * 2 + [[None, None]]
reg.inputs.sampling_percentage = [0.3] * 2 + [[None, None]]
reg.inputs.convergence_threshold = [1.e-7] * 2 + [-0.01]
reg.inputs.convergence_window_size = [20] * 2 + [5]
reg.inputs.smoothing_sigmas = [[4, 2, 1]] * 2 + [[1, 0.5, 0]]
reg.inputs.sigma_units = ['vox'] * 3
reg.inputs.shrink_factors = [[3, 2, 1]]*2 + [[4, 2, 1]]
reg.inputs.use_estimate_learning_rate_once = [True] * 3
reg.inputs.use_histogram_matching = [False] * 2 + [True]
reg.inputs.winsorize_lower_quantile = 0.005
reg.inputs.winsorize_upper_quantile = 0.995
reg.inputs.args = '--float'
reg.inputs.output_warped_image = 'anat2mni_warped_image.nii.gz'
reg.inputs.output_inverse_warped_image = 'mni2anat_warped_image.nii.gz'
reg.inputs.num_threads = nprocs

reg.inputs.fixed_image=template_brain
wf.connect(n4, 'output_image', reg, 'moving_image')

# """
# Transform MNI mask, which is aligned to T1, to MNI
# """

warp_mask = pe.MapNode(ants.ApplyTransforms(), iterfield=['input_image','transforms'], name='warp_mask')
warp_mask.inputs.input_image_type = 0
warp_mask.inputs.interpolation = 'Linear'
warp_mask.inputs.invert_transform_flags = [False]
warp_mask.inputs.input_image = os.path.join(project_dir,'MNI','nihpd_asym_04.5-08.5_mask.nii')


wf.connect(reg, 'inverse_composite_transform', warp_mask, 'transforms') # using transform matrix from t1 to mni
wf.connect(n4, 'output_image', warp_mask, 'reference_image')


# """
# Save data
# """

datasink = pe.Node(nio.DataSink(), name='sinker')
datasink.inputs.base_directory = os.path.join(project_dir, "reg_n4")

datasink.inputs.container = subject_id+'_'+scan_id

wf.connect(reg, 'warped_image', datasink, 'anat.anat2mni')
wf.connect(reg, 'inverse_warped_image', datasink, 'anat.mni2anat')
wf.connect(reg, 'composite_transform', datasink, 'anat.anat2mni_mat')
wf.connect(reg, 'inverse_composite_transform', datasink, 'anat.mni2anat_mat')
wf.connect(warp_mask, 'output_image', datasink, 'anat.mask2t1')

# """
# Run
# """
outgraph = wf.run(plugin='MultiProc')

print("--- %s seconds ---" % (time.time() - start_time))