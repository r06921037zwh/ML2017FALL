#!/bin/bash
wget https://github.com/b02901156/ML-HW3-Training-ED/releases/download/ver2/hw3_model.h5
python3 hw3_test.py $1 $2
