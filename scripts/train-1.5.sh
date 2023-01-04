#!/usr/bin/env bash

versions='NoReg RigidReg RigidAffineReg'

for i in 0 1 2 3 4
do
    for version in $versions
    do
        python train.py "$version""_f""$i" 1.5
    done
done