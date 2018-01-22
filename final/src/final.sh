#!/bin/bash
wget https://github.com/b02901156/ML-HW3-Training-ED/releases/download/ver1/model.zip
unzip model.zip
python3 final.py $1 $2
