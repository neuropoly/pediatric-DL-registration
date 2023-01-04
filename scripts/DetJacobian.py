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

def CalculateDetJacobian(ddf):
    # Part of Jacobian calculations (lines 1595 to 1615) inspired from: https://github.com/dykuang/Medical-image-registration/blob/master/source/Utils.py

    height, width, depth, num_channel = ddf.shape

    num_voxel = (height-4)*(width-4)*(depth-4)

    dx = np.reshape((ddf[:-4,2:-2,2:-2,:]-8*ddf[1:-3,2:-2,2:-2,:] + 8*ddf[3:-1,2:-2,2:-2,:] - ddf[4:,2:-2,2:-2,:])/12.0, [num_voxel, num_channel])
    dy = np.reshape((ddf[2:-2,:-4,2:-2,:]-8*ddf[2:-2,1:-3,2:-2,:] + 8*ddf[2:-2,3:-1,2:-2,:] - ddf[2:-2,4:,2:-2,:])/12.0, [num_voxel, num_channel])
    dz = np.reshape((ddf[2:-2,2:-2,:-4,:]-8*ddf[2:-2,2:-2,1:-3,:] + 8*ddf[2:-2,2:-2,3:-1,:] - ddf[2:-2,2:-2,4:,:])/12.0, [num_voxel, num_channel])
    J = np.stack([dx, dy, dz], 2)

    # Add 1 to diagonal elements
    J[:,0,0] = J[:,0,0]+1
    J[:,1,1] = J[:,1,1]+1
    J[:,2,2] = J[:,2,2]+1

    # calculate determinant for each 3x3 matrix per voxel
    det = np.linalg.det(J)

    return det


def DetJacobianPair():

    version = str(sys.argv[1]) #e.g. "RigidReg"
    time_stamp = str(sys.argv[2])#e.g. "20210527-222928_f0"
    res = str(sys.argv[3])#e.g. Resolution 1.5, 2,0 (in mm)

    fold = time_stamp[-2:] # equivalent to fo, f1 to f4

    start = timer()

    pair_number = sorted(os.listdir(f"/media/andjela/SeagatePor/logs_predict_{version}_{res}/{time_stamp}/test/"))
    remove_list=[]
    for i in pair_number:
        if i.endswith('.csv'):
            remove_list.append(i)
    for element in remove_list:
        if element in pair_number:
            pair_number.remove(element)

    det_Ja_average = []
    det_Ja_std = []
    pairs = []
    percentages = []
    negatives_dets = []

    for pair in pair_number:

        ddf_path = f"/media/andjela/SeagatePor/logs_predict_{version}_{res}/{time_stamp}/test/{pair}/ddf.nii.gz"
        ddf = np.array(nib.load(ddf_path).get_fdata())

        height, width, depth, num_channel = ddf.shape

        det = CalculateDetJacobian(ddf)

        comparison = np.where(np.array(det) > 0, 0, np.array(det)) #all negatives values are there and positives become 0
        negative_dets = np.count_nonzero(comparison)
        percentage_negative = 100 * (negative_dets/len(det))
        #save all determinant jacobian values
        # data = {'det' : det }
        # df_1 = pd.DataFrame(data=data)
        # df_1.to_csv(f"/media/andjela/SeagatePor/other_metrics/Det_{pair}_{version[:-5]}_{time_stamp}")

        percentages.append(percentage_negative)
        det_Ja_average.append(np.mean(det))
        det_Ja_std.append(np.std(det))
        pairs.append(pair)
        negatives_dets.append(negative_dets)
        det_Ja = np.reshape(np.array(det), [height-4, width-4, depth-4,1])
        # det_Ja = np.reshape(np.array(det), [height-1, width-1, depth-1,1])
        # print(det_Ja.shape)

        img = nib.Nifti1Image(det_Ja, None)
        nib.save(img, f"/media/andjela/SeagatePor/logs_predict_{version}_{res}/{time_stamp}/test/{pair}/det_Ja.nii.gz")

    # print(len(det))
    d = {'pair_number' : pairs , 'Det_Ja average': det_Ja_average, 'Det_Ja std': det_Ja_std, 'Count of Negatives':negatives_dets, 'Percentage of Negatives':percentages}
    df = pd.DataFrame(data=d)

    if not os.path.exists(f"/media/andjela/SeagatePor/other_metrics/DetJa_{version}_{fold}_{res}/"):
        os.mkdir(f"/media/andjela/SeagatePor/other_metrics/DetJa_{version}_{fold}_{res}/")

    df.to_csv(f"/media/andjela/SeagatePor/other_metrics/DetJa_{version}_{fold}_{res}/Det_Ja_{time_stamp}")

    end = timer()
    print(f'Time Taken DetJa {version}_{fold}:', timedelta(seconds=end-start))

def DetJacobianRegionsPair():
    version = str(sys.argv[1]) #e.g. "RigidReg"
    time_stamp = str(sys.argv[2])#e.g. "20210527-222928_f0 {fold}"
    res = str(sys.argv[3])#e.g. 1.5, 2.0

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

        det_Ja_average = []
        det_Ja_std = []
        pairs = []
        percentages =[]
        negatives_dets = []
        mov_fix_s = []

        for pair_number in pair_numbers:

            idx = [int(s) for s in pair_number.split("_") if s.isdigit()]

            p, mov, fix = FindFolderNamePair(folder_names[idx[0]])

            fixed_path = f"/media/andjela/SeagatePor/dataset_{version}_{res}_{fold}/test/labels/{p}_{mov}_{fix}/{fix}_{seg}_bin.nii.gz"
            ddf_path = f"/media/andjela/SeagatePor/logs_predict_{version}_{res}/{time_stamp}/test/{pair_number}/ddf.nii.gz"



            fixed_seg_bin = nib.load(fixed_path).get_fdata()

            dim_1 = fixed_seg_bin * nib.load(ddf_path).get_fdata()[:,:,:,0]
            dim_2 = fixed_seg_bin * nib.load(ddf_path).get_fdata()[:,:,:,1]
            dim_3 = fixed_seg_bin * nib.load(ddf_path).get_fdata()[:,:,:,2]

            new_ddf = np.stack((dim_1,dim_2,dim_3),axis=3)
            height, width, depth, num_channel = new_ddf.shape

            det = CalculateDetJacobian(new_ddf)

            comparison = np.where(np.array(det) > 0, 0, np.array(det)) #all negatives values are there and positives become 0
            negative_dets = np.count_nonzero(comparison)
            percentage_negative = 100 * (negative_dets/len(det))

            percentages.append(percentage_negative)
            negatives_dets.append(negative_dets)
            det_Ja_average.append(np.mean(det))
            det_Ja_std.append(np.std(det))
            pairs.append(pair_number)
            mov_fix_s.append(f'{mov}_{fix}')

            det_Ja = np.reshape(np.array(det), [height-4, width-4, depth-4,1])

            img = nib.Nifti1Image(det_Ja, None)
            nib.save(img, f"/media/andjela/SeagatePor/logs_predict_{version}_{res}/{time_stamp}/test/{pair_number}/det_Ja_{seg}.nii.gz")


        d = {'pair_number' : pairs , 'scanIDs' : mov_fix_s , 'Det_Ja average': det_Ja_average, 'Det_Ja std': det_Ja_std, 'Count of Negatives':negatives_dets, 'Percentage of Negatives':percentages}
        df = pd.DataFrame(data=d)
        if not os.path.exists(f"/media/andjela/SeagatePor/other_metrics/DetJa_{version}_{fold}_{res}/"):
            os.mkdir(f"/media/andjela/SeagatePor/other_metrics/DetJa_{version}_{fold}_{res}/")

        df.to_csv(f"/media/andjela/SeagatePor/other_metrics/DetJa_{version}_{fold}_{res}/Det_Ja_{seg}_{time_stamp}")

        end = timer()
        print(f'Time Taken DetJaRegions {version}_{fold}:', timedelta(seconds=end-start))

if __name__ == '__main__':
    
    DetJacobianPair()
    DetJacobianRegionsPair()