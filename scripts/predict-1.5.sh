#!/usr/bin/env bash

# Model Versions
versions='NoReg RigidReg RigidAffineReg'
# Indicate checkpoint of training to be used for predicting
ckpt='250'
# Indicate dates of training to be used for each 5 folds/model
datesNR="20221028-151623 20221030-142912 20221101-135725 20221103-115301 20221105-090652"
datesRR="20221029-070017 20221031-061627 20221102-054412 20221104-025708 20221106-001759"
datesRAR="20221029-224446 20221031-220234 20221102-205001 20221104-180115 20221106-142858"

for version in $versions; do
    count=0
    if [ "$version" = 'NoReg' ]; then
        for dateNR in $datesNR; do
            python predict.py "$dateNR" "$ckpt" "$version""_f""$count" "1.5" >> times_NR.txt
            count=$(($count+1))
        done
    count=0
    elif [ "$version" = 'RigidReg' ]; then
        for dateRR in $datesRR; do
            python predict.py "$dateRR" "$ckpt" "$version""_f""$count" "1.5" >> times_RR.txt
            count=$(($count+1))
        done
    count=0
    else
        for dateRAR in $datesRAR; do
            python predict.py "$dateRAR" "$ckpt" "$version""_f""$count" "1.5" >> times_RAR.txt
            count=$(($count+1))
        done
    fi

done