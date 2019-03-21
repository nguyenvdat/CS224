#!/bin/bash
pip install numpy --upgrade
pip install -q torch==1.0.0 torchvision
git clone https://github.com/nguyenvdat/CS224.git
cd CS224
git checkout a5
cd a5
sh run.sh vocab
pip install -r gpu_requirements.txt
