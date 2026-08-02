#!/bin/bash
set -e

echo "=== CTranslate2 CUDA Compilation for Jetson Orin ==="
echo "Installing build prerequisites..."
sudo apt-get update
sudo apt-get install -y build-essential cmake git libgoogle-perftools-dev

# Navigate to temporary directory inside workspace for compilation
cd /home/ilya/PycharmProjects/VideoTranslator
rm -rf CTranslate2_source
mkdir CTranslate2_source
cd CTranslate2_source

echo "--> Cloning CTranslate2 repository..."
git clone --recursive https://github.com/OpenNMT/CTranslate2.git .

echo "--> Creating build directory..."
mkdir build && cd build

echo "--> Running CMake (CUDA=ON, cuDNN=ON, Jetson Orin Arch=87)..."
cmake .. \
  -DWITH_CUDA=ON \
  -DWITH_CUDNN=ON \
  -DWITH_MKL=OFF \
  -DOPENMP_RUNTIME=COMP \
  -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DCMAKE_INSTALL_PREFIX=/usr/local

echo "--> Compiling CTranslate2 core (using $(nproc) cores)..."
make -j$(nproc)

echo "--> Installing CTranslate2 core libraries..."
sudo make install
sudo ldconfig

echo "--> Installing CTranslate2 Python bindings into virtual environment..."
cd ../python
/home/ilya/PycharmProjects/VideoTranslator/.venv/bin/pip install -r install_requirements.txt
/home/ilya/PycharmProjects/VideoTranslator/.venv/bin/pip install --force-reinstall --no-cache-dir .

echo "--> Cleaning up temporary compilation directory..."
cd /home/ilya/PycharmProjects/VideoTranslator
rm -rf CTranslate2_source

echo "=== Done! CTranslate2 compiled and installed with GPU support successfully. ==="
