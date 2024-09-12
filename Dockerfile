# Deploy on jetson-jetpack4

FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

ENV DEBIAN_FRONTEND=noninteractive

# Install linux packages
# g++ required to build 'tflite_support' package
RUN apt update \
    && apt install --no-install-recommendsx libglib2.0-0 libpython3-dev gnupg g++
# RUN alias python=python3

# Clone YOLOv5 repository
RUN git clone https://github.com/ultralytics/yolov5 /yolov5 

# Create working directory
WORKDIR /yolov5

# Copy contents
RUN git clone https://github.com/DaKhanh/pruning_distillation_yolov5

# Install pip packages manually for TensorRT compatibility https://github.com/NVIDIA/TensorRT/issues/2567
RUN python3 -m pip install --upgrade pip wheel
RUN pip install --no-cache tqdm matplotlib pyyaml psutil thop pandas onnx "numpy==1.23"
RUN pip install --no-cache -e . --no-deps

# Set environment variables
ENV OMP_NUM_THREADS=1


# Usage Examples -------------------------------------------------------------------------------------------------------

# Pull and Run
# t=ultralytics/ultralytics:latest-jetson-jetpack4 && sudo docker pull $t && sudo docker run -it --ipc=host --runtime=nvidia $t
