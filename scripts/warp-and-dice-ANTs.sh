#!/usr/bin/env bash

# Model Versions and transfos
versions='NoReg RigidReg RigidAffineReg'
transfos='rigid-affine-syn-1.5 affine-syn-1.5 syn-1.5'
# Indicate dates of predictions to be used for each 5 folds/model
datesNR="20221109-151016 20221109-151643 20221109-152228 20221109-152826 20221109-153428"
datesRR="20221109-154022 20221109-154546 20221109-155226 20221109-155842 20221109-160526"
datesRAR="20221109-161357 20221109-162138 20221109-163615 20221109-164355 20221109-165110"

for version in $versions; do
    for transfo in $transfos; do
        count=0
        if [ "$version" = 'NoReg' ] && [ "$transfo" = 'rigid-affine-syn-1.5' ]; then
            for dateNR in $datesNR; do
                python WarpAndDice.py "$version" "$dateNR""_f""$count" "1.5" "$transfo" >> times_WarpAndDice_ANTs_NR.txt
                count=$(($count+1))
            done
        count=0
        elif [ "$version" = 'RigidReg' ] && [ "$transfo" = 'affine-syn-1.5' ]; then
            for dateRR in $datesRR; do
                python WarpAndDice.py "$version" "$dateRR""_f""$count" "1.5" "$transfo" >> times_WarpAndDice_ANTs_RR.txt
                count=$(($count+1))
            done
        count=0
        elif [ "$version" = 'RigidAffineReg' ] && [ "$transfo" = 'syn-1.5' ]; then
            for dateRAR in $datesRAR; do
                python WarpAndDice.py "$version" "$dateRAR""_f""$count" "1.5" "$transfo" >> times_WarpAndDice_ANTs_RAR.txt
                count=$(($count+1))
            done
        else
            echo "Unused case"
        fi
    done

done