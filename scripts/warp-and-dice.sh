#!/usr/bin/env bash

# Model Versions
versions='NoReg RigidReg RigidAffineReg'
# Indicate dates of predictions to be used for each 5 folds/model
datesNR="20221109-151016 20221109-151643 20221109-152228 20221109-152826 20221109-153428"
datesRR="20221109-154022 20221109-154546 20221109-155226 20221109-155842 20221109-160526"
datesRAR="20221109-161357 20221109-162138 20221109-163615 20221109-164355 20221109-165110"

for version in $versions; do
    count=0
    if [ "$version" = 'NoReg' ]; then
        for dateNR in $datesNR; do
            python WarpAndDice.py "$version" "$dateNR""_f""$count" "1.5" >> times_WarpAndDice_NR.txt
            count=$(($count+1))
        done
    count=0
    elif [ "$version" = 'RigidReg' ]; then
        for dateRR in $datesRR; do
            python WarpAndDice.py "$version" "$dateRR""_f""$count" "1.5" >> times_WarpAndDice_RR.txt
            count=$(($count+1))
        done
    count=0
    else
        for dateRAR in $datesRAR; do
            python WarpAndDice.py "$version" "$dateRAR""_f""$count" "1.5" >> times_WarpAndDice_RAR.txt
            count=$(($count+1))
        done
    fi

done