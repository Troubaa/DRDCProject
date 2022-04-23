# DRDCProject
Cals Simulator with the A3C Neural Network from the StarCraft 2 environment.

This program requires quite a bit of setup before it can run, and it is very important to make sure 
to download the proper versions. Otherwise conflicts between versions will occur and the 
program will not properly configure or won’t run correctly.

First step is if one has a Nvidia Graphics Processing Unit (GPU) they must setup Nvidia CUDA 
Version 10.0.130 and cuDNN Version 7.6.5. The installation process can be found on the 
Nvidia’s development website. Since, training the neural network requires a lot of computation 
power to calculate all the new weights in the neural network. These programs provided by 
Nvidia ensure that the GPU can be used with the Tensorflow package to reduce the stress on 
one’s central processing unit as calculations can be completed using the GPU as well.
Next are the following python packages required and their respected command needed to 
download the correct version.

pip install tensorflow==1.14

pip install tensorflow-gpu==1.14

pip install absl-py

pip install pysc2==1.1

Then run train_network.py file.