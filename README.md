## Description
This is a PyTorch Re-Implementation of [FOTS: Fast Oriented Text Spotting with a Unified Network](https://arxiv.org/abs/1801.01671).

* Use RBOX part in EAST for text detection.
* Using dice loss and GIOU loss for EAST. Codes mainly refer to [SakuraRiven/EAST](https://github.com/SakuraRiven/EAST)
* The pre-trained model provided only achieves __30.88__ F-score on ICDAR 2015 Challenge 4 using only the 1000 images. see [here](https://rrc.cvc.uab.es/?ch=4&com=evaluation&view=method_info&task=4&m=61516) for the detailed results.The main problem is with word recognition or don't use extra dataset

| Model | Loss | Recall | Precision | F-score | 
| - | - | - | - | - |
| Re-Implement | Dice+GIOU | 34.33 | 32.06 | 33.16 |

## Prerequisites
Only tested on
* Anaconda3
* Python 3.5.6
* PyTorch 0.4.1
* Shapely 1.6.4
* opencv-python 4.0.0.21
* lanms 1.0.2

When running the script, if some module is not installed you will see a notification and installation instructions. __if you failed to install lanms, please update gcc and binutils__. The update under conda environment is:

    conda install -c omgarcia gcc-6
    conda install -c conda-forge binutils

## Main Problem
the main problem is in the recognize branch, maybe can use extra dataset,such as synth800k and others.
