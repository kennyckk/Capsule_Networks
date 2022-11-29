This project is to reproduce the 3 Experimental Results of the paper Dynamic Routing Between Capsules (2017)
by S.Sabour, N.Frosst, Geoffrey E. Hinton

Each of the exp folder contains the .py file which would train and test the corresponding model in each experiment
mentioned in the paper. Data and Pre-trained parameters were also included in the folder. 
If re-training is preferred, please delete the .pt pre-trained parameters file from the directory and run the .py file from scratch

The code is executed on python 3.7 and the follow packages

pytorch 1.12.1+cu113	
matplotlib 3.2.2
scipy 1.7.3
numpy 1.21.6
pandas 1.3.5

Experiment 1: High Performing Digit Classification & Image Reconstruction
Results: 0.4% vs 0.25% (Paper) test error  

Experiment 2: Classify Affined Digits 
Results: 77% vs 79% (Paper) Test Accuracy

Experiment 3: Multi-Digits Segmentation & Classification (Small Training Data used)
Results: 85% vs 95% (Paper) Test Accuracy