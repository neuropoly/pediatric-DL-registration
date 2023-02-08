# -*- coding: utf-8 -*-
# This code was adapted from: https://github.com/dpaniukov/CBF_early_childhood/blob/master/preproc/cbf2mni.py 

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

nprocs = multiprocessing.cpu_count()

start_time = time.time()

fsl.FSLCommand.set_default_output_type('NIFTI_GZ')

"""
Project info
"""

main_dir = "/home/andjela/"
project_dir = main_dir+"Documents/Preprocess/"
work_dir = main_dir+"Documents/work_dir2/"


if not os.path.exists(work_dir):
    os.makedirs(work_dir)

template_brain = os.path.join(project_dir, 'MNI', 'nihpd_asym_04.5-08.5_t1w.nii') 
brain_mask_MNI = os.path.join(project_dir, 'MNI', 'nihpd_asym_04.5-08.5_mask.nii')

# which subjects to run
subject_id = str(sys.argv[1])
scan_id = str(sys.argv[2])


subj_reg = os.path.join(project_dir, "reg_n4")
brain_mask = os.path.join(subj_reg, subject_id + "_" + scan_id + "/anat/mask2t1/_warp_mask0/nihpd_asym_04.5-08.5_mask_trans.nii")
composite_transform = os.path.join(subj_reg, subject_id + "_" + scan_id + "/anat/anat2mni_mat/anat2mni_Composite.h5")


"""
Create workflow
"""

wf = pe.Workflow(name='wf')
wf.base_dir = os.path.join(work_dir, "cbf2mni_wdir", subject_id, scan_id)
wf.config = {"execution": {"crashdump_dir": os.path.join(work_dir, 'cbf2mni_crashdumps', subject_id, scan_id)}}

datasource = pe.Node(nio.DataGrabber(infields=['subject_id', 'scan_id'], outfields=['anat']), name='datasource')
datasource.inputs.base_directory = project_dir
datasource.inputs.template = '*'
datasource.inputs.field_template = dict(anat='OSFData/%s/%s/%s.nii.gz')
datasource.inputs.template_args = dict(anat=[['subject_id', 'scan_id', 'scan_id']])
datasource.inputs.sort_filelist = True
datasource.inputs.subject_id = subject_id
datasource.inputs.scan_id = scan_id

inputnode = pe.Node(interface=util.IdentityInterface(fields=['anat']),name='inputspec')

wf.connect([(datasource, inputnode, [('anat', 'anat') ]), ])



"""
T1 skull stripping
"""
biniraze_mask = pe.Node(interface=fsl.ImageMaths(out_data_type='float',
                                                            op_string='-bin',
                                                            suffix='_dtype'),
                                   iterfield=['in_file'],
                                   name='biniraze_mask')

biniraze_mask.inputs.in_file = brain_mask

brainextraction = pe.Node(interface=fsl.ImageMaths(out_data_type='float',
                                                      op_string='-mul',
                                                      suffix='_dtype'),
                             iterfield=['in_file', 'in_file2'],
                             name='brainextraction')

wf.connect(inputnode, 'anat', brainextraction, 'in_file')
wf.connect(biniraze_mask, 'out_file', brainextraction, 'in_file2')


"""
Save data
"""

datasink = pe.Node(nio.DataSink(), name='sinker')
datasink.inputs.base_directory = os.path.join(project_dir, "cbf2mni")

datasink.inputs.container = subject_id + '_' + scan_id

wf.connect(brainextraction, 'out_file', datasink, 'anat.brain')


"""
Run
"""
outgraph = wf.run(plugin='MultiProc')

print("--- %s seconds ---" % (time.time() - start_time))