# Yolov5
This project focuses on applying pruning technique to compress Yolov5s model, and training this model using Knowledge Distillation from Yolov5x6 teacher model, effectively reducing parameters and training time while preserving accuracy. These omptimizations are essential for deploying on resource-constrained embedded system such as Jetson and Rapberry Pi.

# Setup
1. Download Yolov5
'''bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
'''
2. Download this repo
'''bash
git clone https://github.com/DaKhanh/yolov5_pruning_distillation
'''

# Reference
https://github.com/ultralytics/yolov5
https://github.com/VainF/Torch-Pruning