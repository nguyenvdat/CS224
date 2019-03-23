#!/bin/bash
pip install numpy --upgrade
pip install -q torch==1.0.0 torchvision
sudo apt-get install git-lfs
git lfs install
git clone https://github.com/nguyenvdat/CS224.git
git clone https://github.com/nguyenvdat/model.git
cd model
git checkout CS224
cd ../
cp model/CS224/a5/model.bin CS224/a5
cp model/CS224/a5/model.bin.optim CS224/a5
cd CS224
git checkout a5
cd a5
sh run.sh vocab
pip install -r gpu_requirements.txt
