import os
import sys
import shutil
import math
import time
import h5py
from collections import Counter
from timeit import default_timer as timer
from datetime import timedelta
import numpy as np
import nibabel as nib
import tkinter
import pandas as pd
# # To uncomment the following for dice scores on predictions, not used for DiceScoreANTs
# from deepreg.warp import warp

def FindFolderNamePair(name):
    #Outputs patient number, moving and fixed image scanID as strings for further analysis
    #Possible folder name pairs are below with each string differing in length
    # name = '101117_CL_Dev_004_CL_Dev_008'
    # name1 = '101117_CL_Dev_004_PS15_048'
    # name2 = '101117_PS15_048_CL_Dev_004'
    # name3 = '10097_PS15_048_PS17_017'

    #idx contains a list of strings of a given folder name
    idx = [s for s in name.split("_")]

    if len(idx) == 6:
        if 'CL' in idx[1]:
            mov = f'{idx[1]}_{idx[2]}_{idx[3]}'
            fix = f'{idx[4]}_{idx[5]}'
            p = f'{idx[0]}'
            return(p, mov, fix)
        elif 'PS' in idx[1]:
            mov = f'{idx[1]}_{idx[2]}'
            fix = f'{idx[3]}_{idx[4]}_{idx[5]}'
            p = f'{idx[0]}'
            return(p, mov, fix)

    elif len(idx) == 7:
        mov = f'{idx[1]}_{idx[2]}_{idx[3]}'
        fix = f'{idx[4]}_{idx[5]}_{idx[6]}'
        p = f'{idx[0]}'
        return(p, mov, fix)

    elif len(idx) == 5:
        mov = f'{idx[1]}_{idx[2]}'
        fix = f'{idx[3]}_{idx[4]}'
        p = f'{idx[0]}'
        return(p, mov, fix)

    elif len(idx) == 4:
        mov = f'{idx[1]}'
        fix = f'{idx[2]}_{idx[3]}'
        p = f'{idx[0]}'
        return(p, mov, fix)


    else:
        print('Not a corresponding folder name')

def WarpSegsPair(version, time_stamp, res):

    # version = str(sys.argv[1]) #e.g. "RigidReg"
    # time_stamp = str(sys.argv[2])#e.g. "20210527-222928_f0" time-stamp of predictions
    # res = str(sys.argv[3])#e.g. 1.5, 2.0 etc

    start = timer()

    segs = ['wm','gm','csf']

    fold = time_stamp[-2:] # equivalent to fo, f1 to f4

    for seg in segs:

        pair_numbers = sorted(os.listdir(f"/media/andjela/SeagatePor/logs_predict_{version}_{res}/{time_stamp}/test/"))
        remove_list=[] #remove .csv files in test folder
        for i in pair_numbers:
            if i.endswith('.csv'):
                remove_list.append(i)
        for element in remove_list:
            if element in pair_numbers:
                pair_numbers.remove(element)

        folder_names = sorted(os.listdir(f"/media/andjela/SeagatePor/dataset_{version}_{res}_{fold}/test/labels/"))

        for pair_number in pair_numbers:
            idx = [int(s) for s in pair_number.split("_") if s.isdigit()]

            p, mov, fix = FindFolderNamePair(folder_names[idx[0]])

            image_path = f"/media/andjela/SeagatePor/dataset_{version}_{res}_{fold}/test/labels/{p}_{mov}_{fix}/{mov}_{seg}.nii.gz"
            ddf_path = f"/media/andjela/SeagatePor/logs_predict_{version}_{res}/{time_stamp}/test/{pair_number}/ddf.nii.gz"

            if not os.path.exists(f"/media/andjela/SeagatePor/warped_segs/{version}_{res}/"):
                os.mkdir(f"/media/andjela/SeagatePor/warped_segs/{version}_{res}/")
            out_path = f"/media/andjela/SeagatePor/warped_segs/{version}_{res}/warped_{pair_number}_{seg}.nii.gz"
            warp(image_path, ddf_path, out_path)


    end = timer()
    print(f'Time Taken Warp {version}_{fold}:', timedelta(seconds=end-start))

def BinarizeSegs(wm_path, gm_path, csf_path):


    wm_img = nib.load(wm_path)
    gm_img = nib.load(gm_path)
    csf_img = nib.load(csf_path)

    #Keep a copy of header information for saving uses later on
    hdr_wm =  wm_img.header.copy()
    hdr_gm =  gm_img.header.copy()
    hdr_csf =  csf_img.header.copy()

    wm =  wm_img.get_fdata()
    gm =  gm_img.get_fdata()
    csf =  csf_img.get_fdata()

    height, width, depth =  wm.shape
    num_voxel = height * width * depth

    #Create 1D arrays from 3D segmentation probabilities
    wm = np.reshape( wm, [num_voxel, 1])
    gm = np.reshape( gm, [num_voxel, 1])
    csf = np.reshape( csf, [num_voxel, 1])

    #Stack all segmentations
    segs = np.stack([wm,gm,csf],1)
    # print( 'segs', segs.shape)

    # Find max probability in third dimension, outside of brain region max_prob=0
    max_prob = np.amax(segs, axis=1)
    # print('max', max_prob.shape, max_prob[0])
    # Bring every pixel where prob<0.03 or prob=0 to 10 to be easily identifiable afterward for changes
    max_prob = np.where((max_prob<0.03)|(max_prob==0),10,max_prob)

    # Find index (region) (-> 0 for wm, 1 for gm and 2 for csf) where there is the max probability

    max_region = np.argmax(segs,axis=1)
    # print('max', max_region.shape, max_region[0:7])

    # Binarize each image by putting to 1 where the probability is maximum and 0 elsewhere
    # Then remove where the max_probability was equal 0 or was smaller than 0.03
    bin_wm = np.where((max_region==0),1,0)
    bin_wm = np.where((max_prob==10),0,bin_wm)
    bin_gm = np.where((max_region==1),1,0)
    bin_gm = np.where((max_prob==10),0,bin_gm)
    bin_csf = np.where((max_region==2),1,0)
    bin_csf = np.where((max_prob==10),0,bin_csf)

    #Reshape binarized versions to 3D image
    bin_wm = np.reshape(bin_wm,[height, width, depth])
    bin_gm = np.reshape(bin_gm,[height, width, depth])
    bin_csf = np.reshape(bin_csf,[height, width, depth])

    return bin_wm, bin_gm, bin_csf, hdr_wm, hdr_gm, hdr_csf


def BinarizeWarpedSegs(version, res, pair):
    # Binarize warped segmentations for a specific pair

    wm_path = f"/media/andjela/SeagatePor/warped_segs/{version}_{res}/warped_{pair}_wm.nii.gz"
    gm_path = f"/media/andjela/SeagatePor/warped_segs/{version}_{res}/warped_{pair}_gm.nii.gz"
    csf_path = f"/media/andjela/SeagatePor/warped_segs/{version}_{res}/warped_{pair}_csf.nii.gz"

    bin_wm, bin_gm, bin_csf, hdr_wm, hdr_gm, hdr_csf = BinarizeSegs(wm_path, gm_path, csf_path)

    # Save warped binarized versions
    img_wm = nib.Nifti1Image(bin_wm, None,hdr_wm)
    nib.save(img_wm, f"/media/andjela/SeagatePor/warped_segs/{version}_{res}/warped_{pair}_wm_bin.nii.gz")
    img_gm = nib.Nifti1Image(bin_gm, None,hdr_gm)
    nib.save(img_gm, f"/media/andjela/SeagatePor/warped_segs/{version}_{res}/warped_{pair}_gm_bin.nii.gz")
    img_csf = nib.Nifti1Image(bin_csf, None,hdr_csf)
    nib.save(img_csf, f"/media/andjela/SeagatePor/warped_segs/{version}_{res}/warped_{pair}_csf_bin.nii.gz")

def BinarizeWarpedSegsANTs(version, res, pair):
    # Binarize warped segmentations for a specific pair

    wm_path = f"/media/andjela/SeagatePor/warped_segs/ANTs/{version}_{res}/warped_{pair}_wm.nii.gz"
    gm_path = f"/media/andjela/SeagatePor/warped_segs/ANTs/{version}_{res}/warped_{pair}_gm.nii.gz"
    csf_path = f"/media/andjela/SeagatePor/warped_segs/ANTs/{version}_{res}/warped_{pair}_csf.nii.gz"

    bin_wm, bin_gm, bin_csf, hdr_wm, hdr_gm, hdr_csf = BinarizeSegs(wm_path, gm_path, csf_path)

    # Save warped binarized versions
    img_wm = nib.Nifti1Image(bin_wm, None,hdr_wm)
    nib.save(img_wm, f"/media/andjela/SeagatePor/warped_segs/ANTs/{version}_{res}/warped_{pair}_wm_bin.nii.gz")
    img_gm = nib.Nifti1Image(bin_gm, None,hdr_gm)
    nib.save(img_gm, f"/media/andjela/SeagatePor/warped_segs/ANTs/{version}_{res}/warped_{pair}_gm_bin.nii.gz")
    img_csf = nib.Nifti1Image(bin_csf, None,hdr_csf)
    nib.save(img_csf, f"/media/andjela/SeagatePor/warped_segs/ANTs/{version}_{res}/warped_{pair}_csf_bin.nii.gz")


def BinarizeInitialSegs(version, res, fold, p, mov, fix):
    # Binarize each mov, fix segmentations


    fix_wm_path = f"/media/andjela/SeagatePor/dataset_{version}_{res}_{fold}/test/labels/{p}_{mov}_{fix}/{fix}_wm.nii.gz"
    fix_gm_path = f"/media/andjela/SeagatePor/dataset_{version}_{res}_{fold}/test/labels/{p}_{mov}_{fix}/{fix}_gm.nii.gz"
    fix_csf_path = f"/media/andjela/SeagatePor/dataset_{version}_{res}_{fold}/test/labels/{p}_{mov}_{fix}/{fix}_csf.nii.gz"
    mov_wm_path = f"/media/andjela/SeagatePor/dataset_{version}_{res}_{fold}/test/labels/{p}_{mov}_{fix}/{mov}_wm.nii.gz"
    mov_gm_path = f"/media/andjela/SeagatePor/dataset_{version}_{res}_{fold}/test/labels/{p}_{mov}_{fix}/{mov}_gm.nii.gz"
    mov_csf_path = f"/media/andjela/SeagatePor/dataset_{version}_{res}_{fold}/test/labels/{p}_{mov}_{fix}/{mov}_csf.nii.gz"


    fix_bin_wm, fix_bin_gm, fix_bin_csf, fix_hdr_wm, fix_hdr_gm, fix_hdr_csf = BinarizeSegs(fix_wm_path, fix_gm_path, fix_csf_path)
    mov_bin_wm, mov_bin_gm, mov_bin_csf, mov_hdr_wm, mov_hdr_gm, mov_hdr_csf = BinarizeSegs(mov_wm_path, mov_gm_path, mov_csf_path)

    img_wm = nib.Nifti1Image(fix_bin_wm, None,fix_hdr_wm)
    nib.save(img_wm, f"/media/andjela/SeagatePor/dataset_{version}_{res}_{fold}/test/labels/{p}_{mov}_{fix}/{fix}_wm_bin.nii.gz")
    img_gm = nib.Nifti1Image(fix_bin_gm, None,fix_hdr_gm)
    nib.save(img_gm, f"/media/andjela/SeagatePor/dataset_{version}_{res}_{fold}/test/labels/{p}_{mov}_{fix}/{fix}_gm_bin.nii.gz")
    img_csf = nib.Nifti1Image(fix_bin_csf, None,fix_hdr_csf)
    nib.save(img_csf, f"/media/andjela/SeagatePor/dataset_{version}_{res}_{fold}/test/labels/{p}_{mov}_{fix}/{fix}_csf_bin.nii.gz")
    img_wm = nib.Nifti1Image(mov_bin_wm, None,mov_hdr_wm)
    nib.save(img_wm, f"/media/andjela/SeagatePor/dataset_{version}_{res}_{fold}/test/labels/{p}_{mov}_{fix}/{mov}_wm_bin.nii.gz")
    img_gm = nib.Nifti1Image(mov_bin_gm, None, mov_hdr_gm)
    nib.save(img_gm, f"/media/andjela/SeagatePor/dataset_{version}_{res}_{fold}/test/labels/{p}_{mov}_{fix}/{mov}_gm_bin.nii.gz")
    img_csf = nib.Nifti1Image(mov_bin_csf, None, mov_hdr_csf)
    nib.save(img_csf, f"/media/andjela/SeagatePor/dataset_{version}_{res}_{fold}/test/labels/{p}_{mov}_{fix}/{mov}_csf_bin.nii.gz")



def DiceScoresPair():

    version = str(sys.argv[1]) #e.g. "RigidReg"
    time_stamp = str(sys.argv[2])#e.g. "20210527-222928_f0 {fold}"
    res = str(sys.argv[3])#e.g. Resolution 1.5, 2,0 (in mm)

    WarpSegsPair(version, time_stamp, res)

    start = timer()

    segs = ['wm','gm','csf']

    fold = time_stamp[-2:] # equivalent to fo, f1 to f4

    for seg in segs:

        dice_scores = []
        pairs  = []
        mov_fix_s = []

        pair_numbers = sorted(os.listdir(f"/media/andjela/SeagatePor/logs_predict_{version}_{res}/{time_stamp}/test/"))
        remove_list=[] #remove .csv files in test folder
        for i in pair_numbers:
            if i.endswith('.csv'):
                remove_list.append(i)
        for element in remove_list:
            if element in pair_numbers:
                pair_numbers.remove(element)

        folder_names = sorted(os.listdir(f"/media/andjela/SeagatePor/dataset_{version}_{res}_{fold}/test/labels/"))

        for pair_number in pair_numbers:
            idx = [int(s) for s in pair_number.split("_") if s.isdigit()]

            p, mov, fix = FindFolderNamePair(folder_names[idx[0]])

            BinarizeInitialSegs(version, res, fold, p, mov, fix)
            fixed_seg = nib.load(f"/media/andjela/SeagatePor/dataset_{version}_{res}_{fold}/test/labels/{p}_{mov}_{fix}/{fix}_{seg}_bin.nii.gz").get_fdata()

            BinarizeWarpedSegs(version, res, pair_number)
            warped_seg = nib.load(f"/media/andjela/SeagatePor/warped_segs/{version}_{res}/warped_{pair_number}_{seg}_bin.nii.gz").get_fdata()

            y_true = fixed_seg
            y_pred = warped_seg

            T = (y_true.flatten()>0)
            P = (y_pred.flatten()>0)

            dice_scores.append(2*np.sum(T*P)/(np.sum(T) + np.sum(P)))
            pairs.append(pair_number)
            mov_fix_s.append(f'{mov}_{fix}')

        d = {'pair_number' : pairs , 'scanIDs' : mov_fix_s ,  'Dice Score': dice_scores} 
        df = pd.DataFrame(data=d)

        if not os.path.exists(f"/media/andjela/SeagatePor/dice_scores/{version}_{res}_{fold}/"):
            os.mkdir(f"/media/andjela/SeagatePor/dice_scores/{version}_{res}_{fold}/")

        df.to_csv(f"/media/andjela/SeagatePor/dice_scores/{version}_{res}_{fold}/dice_scores_{version}_{seg}_bin")

    end = timer()
    print(f'Time Taken Dice {version}_{fold}:', timedelta(seconds=end-start))

def WarpSegsANTs(version, time_stamp, res, transfo):
    # Function to warp specific segs using ANTs obtained tranformations
    version = str(sys.argv[1]) #e.g. "RigidReg_run1"
    time_stamp = str(sys.argv[2])#e.g. "20210527-222928"
    res = str(sys.argv[3])#e.g. Resolution 1.5, 2,0 (in mm)
    transfo = str(sys.argv[4])#e.g. 'syn-1.5'


    start = timer()

    segs = ['wm','gm','csf']
    fold = time_stamp[-2:] # equivalent to fo, f1 to f4

    for seg in segs:

        pair_numbers = sorted(os.listdir(f"/media/andjela/SeagatePor/logs_predict_{version}_{res}/{time_stamp}/test/"))
        remove_list=[] #remove .csv files in test folder
        for i in pair_numbers:
            if i.endswith('.csv'):
                remove_list.append(i)
        for element in remove_list:
            if element in pair_numbers:
                pair_numbers.remove(element)

        folder_names = sorted(os.listdir(f"/media/andjela/SeagatePor/dataset_{version}_{res}_{fold}/test/labels/"))

        for pair_number in pair_numbers:
            idx = [int(s) for s in pair_number.split("_") if s.isdigit()]

            p, mov, fix = FindFolderNamePair(folder_names[idx[0]])

            fix_path = f"/media/andjela/SeagatePor/dataset_{version}_{res}_{fold}/test/labels/{p}_{mov}_{fix}/{fix}_{seg}_bin.nii.gz"
            fixed_seg = nib.load(fix_path).get_fdata()

            # Warp images with obtained DDF
            mov_path = f"/media/andjela/SeagatePor/dataset_{version}_{res}_{fold}/test/labels/{p}_{mov}_{fix}/{mov}_{seg}_bin.nii.gz"
            transform_path = f"/media/andjela/SeagatePor/PairReg/{transfo}/images/{p}_{mov}_{fix}/mov2fix_Composite.h5"

            if not os.path.exists(f"/media/andjela/SeagatePor/warped_segs/ANTs/{version}_{res}/"):
                os.mkdir(f"/media/andjela/SeagatePor/warped_segs/ANTs/{version}_{res}/")
            out_path = f"/media/andjela/SeagatePor/warped_segs/ANTs/{version}_{res}/warped_{pair_number}_{seg}.nii.gz"

            os.system(f"antsApplyTransforms --default-value 0 --float 0 " \
                f"--input {mov_path} " \
                f"--input-image-type 0 --interpolation Linear " \
                f"--output {out_path} " \
                f"--reference-image {fix_path} " \
                f"--verbose 1 " \
                f"--transform {transform_path}"
                )

    end = timer()
    print(f'Time Taken Warp {transfo}:', timedelta(seconds=end-start))

def DiceScoresANTs():

    version = str(sys.argv[1]) #e.g. "RigidReg_run1"
    time_stamp = str(sys.argv[2])#e.g. "20210527-222928"
    res = str(sys.argv[3])#e.g. Resolution 1.5, 2,0 (in mm)
    transfo = str(sys.argv[4])#e.g. 'syn-1.5'

    WarpSegsANTs(version, time_stamp, res, transfo)

    start = timer()

    segs = ['wm','gm','csf']
    fold = time_stamp[-2:] # equivalent to fo, f1 to f4

    for seg in segs:

        dice_scores = []
        pairs  = []
        mov_fix_s = []

        pair_numbers = sorted(os.listdir(f"/media/andjela/SeagatePor/logs_predict_{version}_{res}/{time_stamp}/test/"))
        remove_list=[] #remove .csv files in test folder
        for i in pair_numbers:
            if i.endswith('.csv'):
                remove_list.append(i)
        for element in remove_list:
            if element in pair_numbers:
                pair_numbers.remove(element)

        folder_names = sorted(os.listdir(f"/media/andjela/SeagatePor/dataset_{version}_{res}_{fold}/test/labels/"))

        for pair_number in pair_numbers:
            idx = [int(s) for s in pair_number.split("_") if s.isdigit()]

            p, mov, fix = FindFolderNamePair(folder_names[idx[0]])

            
            fix_path = f"/media/andjela/SeagatePor/dataset_{version}_{res}_{fold}/test/labels/{p}_{mov}_{fix}/{fix}_{seg}_bin.nii.gz"
            fixed_seg = nib.load(fix_path).get_fdata()


            BinarizeWarpedSegsANTs(version, res, pair_number)
            warped_seg = nib.load(f"/media/andjela/SeagatePor/warped_segs/ANTs/{version}_{res}/warped_{pair_number}_{seg}_bin.nii.gz").get_fdata()

            y_true = fixed_seg
            y_pred = warped_seg

            T = (y_true.flatten()>0)
            P = (y_pred.flatten()>0)

            dice_scores.append(2*np.sum(T*P)/(np.sum(T) + np.sum(P)))
            pairs.append(pair_number)
            mov_fix_s.append(f'{mov}_{fix}')

        d = {'pair_number' : pairs , 'scanIDs' : mov_fix_s ,  'Dice Score': dice_scores} #f'Average for {version[-5:]}': np.mean(dice_scores)
        df = pd.DataFrame(data=d)

        if not os.path.exists(f"/media/andjela/SeagatePor/dice_scores/ANTs/{version}_{res}_{fold}/"):
            os.mkdir(f"/media/andjela/SeagatePor/dice_scores/ANTs/{version}_{res}_{fold}/")

        df.to_csv(f"/media/andjela/SeagatePor/dice_scores/ANTs/{version}_{res}_{fold}/dice_scores_{version}_{seg}_bin")

    end = timer()
    print(f'Time Taken Dice {transfo}:', timedelta(seconds=end-start))


if __name__ == '__main__':
    
    # For warp-and-dice.sh
    # DiceScoresPair()
    # For warp-and-dice-ANTs.sh
    DiceScoresANTs()