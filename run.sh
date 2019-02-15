#!/bin/bash
if [ "$1" == "1" ]
then
	python3 linear_regression/linear_regression.py $2 $3 $4 $5
elif [ "$1" == "2" ]
then
	python3 locally_weighted_lin_reg/locally_weighted_lin_reg.py $2 $3 $4
elif [ "$1" == "3" ]
then
	python3 logistic_regression/logistic_regression.py $2 $3
elif [ "$1" == "4" ]
then
	python3 gaussian_discriminant_analysis/gaussian_driscriminant_analysis.py $2 $3 $4
fi
