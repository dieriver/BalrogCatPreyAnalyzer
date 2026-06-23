# WARNING: This repository has been archived and moved to https://codeberg.org/dieriver/BalrogCatPreyAnalyzer

# Introduction
If you own a cat that has the freedom to go outside, then you probably are familiar with the issue of your feline
bringing home prey. Balrog aims to perform Cat Prey Detection with Deep Learning on any cat in any environment.
This project was based on the original [Cat_Prey_Analyzer repository](https://github.com/niciBume/Cat_Prey_Analyzer).
This project improves the original software by:

* Factorized the code and organize it in a python package.
* Dropped the dependency on the Raspberry PI camera.
* Interfaced the Surepet cat flap.
* Added telegram commands to control the Surepet cat flap.
* Added general configuration in a file.
* Support for multi-threading analysis of the camera frames.

For detailed information about how the Deep Learning part works, please check the readme on the [original repository](https://github.com/niciBume/Cat_Prey_Analyzer)


# Requirements
## Dependencies
As stated by the original repo, this software relies on tensorflow object detection API that need to be downloaded
separately. Before doing this, we need to install some packages (in Debian-based systems):

```shell
sudo apt install libglu1-mesa-dev libglx-mesa0
```

## Python virtual environment
Please note that Balrog Cat Prey Analyzer depends *on python 3.11*, and python 3.12 is not supported. To fulfill this,
please run (in Debian-based systems):

```shell
sudo apt-get install python3.11-full
```

To install all dependencies, it is recommended to create a python virtual environment (using the python3.11-venv
package which should have been installed with the command below):

```shell
pwd # This will show the current path; we assume this returned '/path/to'
python3.11 -m venv virt-env
```

which will create a python virtual environment under the `virt-env` folder. To activate the virtual environment, simply
use the following command:

```shell
source /path/to/virt-env/bin/activate
(virt-env) $
```

This will allow to install python packages without messing with the python installation on your machine.


## Protobuf compiler
The Tensorflow Object Detection APU also requires a rather old version of the protobuf compiler to work (v3.19.0). To
install this version, please go to the [asset page](https://github.com/protocolbuffers/protobuf/releases/tag/v3.19.0)
of that release, and download the build for your architecture.
For x86_64, you can simply run:

```shell
wget https://github.com/protocolbuffers/protobuf/releases/download/v3.19.0/protoc-3.19.0-linux-x86_64.zip
```

To install it, we can do it in the `.local` folder by simply unzipping that package in the target directory:

```shell
unzip protoc-3.19.0-linux-x86_64.zip -d ~/.local
```

After doing this, please check that the `protoc` compiler is available on your shell by running `protoc --version`
which should return the version of libprotoc that you just installer. If the command returns an error (command not
found), please make sure that the folder `~/.local/bin` is in your PATH variable, and try again. 


## Tensorflow Object Detection API
To download the tensorflow Object Detection API, we simply run (adapted from the [tensorflow object detection repo](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi)):

```shell
pwd # We use this to check the full path of the tensorflow models base folder (tensorflow1). We use it in the rest of this readme
git clone --depth 1 https://github.com/tensorflow/models.git tf-models
```

We now need to use the protobuf compiler to generate the source of the API using the proto files:

```shell
cd /path/to/tf-models/research
protoc object_detection/protos/*.proto --python_out=.
```

to ease the installation, we copy the `setup.py`:

```shell
cd /path/to/tf-models/research
cp object_detection/packages/tf2/setup.py .
```

IMPORTANT: We need to adapt the tensorflow dependencies of this library. To do so, change the required version of the 
`tf-models-officials`:

```shell
'tf-models-official>=2.5.1',
```

to (adding `2.15.0` as the maximum version supported):

```shell
'tf-models-official>=2.5.1,<=2.15.0',
```

Then, we build a python wheel and install the Tensorflow Object Detection API:
```shell
$ source /path/to/virt-env/bin/activate
(venv) pip install build
(venv) python -m build --wheel
(venv) pip install dist/*.whl
```

## Install Balrog and its dependencies
Finally, we are ready to install this module and its dependencies. To do so, we simply activate the virtual environment,
and run pip to install this package:

```shell
$ source /path/to/virt-env/bin/activate
(virt-env) $ pip install .
```


# Configuration
## Environment variables
Balrog uses a few environment variables to configure the interfaces: Camera input, Telegram Bot and Surepet login
details. To configure this, you need to execute the following lines in your shell:

```shell
export CAMERA_STREAM_URI=<camera_rstp_url>
export SUREPET_USER=<surepet_user>
export SUREPET_PASSWORD=<surepet-password>
export TELEGRAM_BOT_TOKEN=<telegram_bot_token>
export TELEGRAM_CHAT_ID=<telegram_chat_id>
```

You can add these lines at the end of the `virt-env/bin/activate` file, so these variables are available each time
that you activate the python virtual environment.

Creating the telegram bot and getting its token and chat ID is out of the scope of this readme, but you can google
for that and check some [answers in Stackoverflow](https://stackoverflow.com/questions/32423837/telegram-bot-how-to-get-a-group-chat-id).

### Debugging - Enabling null Camera and Telegram interfaces
There are two extra environment variables can you can set:

```shell
export BALROG_USE_NULL_CAMERA=1
export BALROG_USE_NULL_TELEGRAM=1
```

The first variable will start the module with instances of a camera class that feed the same static image (the file in
`balrog/resources/dbg_casc-jpg`) on each frame. The second variable will use a "null" telegram interface, which simply
discards all the messages and images that you try to send.  These instances might be quite useful when debugging this
module.


## Configuration file
Before executing, you need to create the configuration file. You can use the `config-template.toml` file as a base, and
create the `config.toml` file with its content.

It is recommended to *not modify* the configurations under the `model` section, since they directly control the
sensitivity of the verdicts generated by the tensorflow model.

# Execution
To execute, simply activate your python virtual environment, export the required variables (if needed) and then simply
execute the module:

```shell
$ source /path/to/virt-env/bin/activate
(virt-env) $ python3 -m balrog
```

This repository also contains a start script that you can use in a "production" environment:

```shell
$ source /path/to/virt-env/bin/activate
(virt-env) $ ./balrog.sh
```

This script will start the module, but also restart the module if it fails for some reason. Additionally, you can use
the `balrog-dbg.sh` script to start the module in a similar manner, but using the `-m` option in python to get extra
debugging info from the python interpreter.


# CUDA Support

Since the principal computation is carried by the OpenCV and TensorFlow libraries, this package also offers support for
accelerating the computation using CUDA. To this end, you need to install this package with cuda support:

```shell
pip install .[with-cuda]
```

However, this *will only install tensorflow with CUDA support*. To extend CUDA support for the OpenCV package, you
need to compile the full OpenCV package with CUDA support.


## Compiling OpenCV with CUDA Support

First, be sure to install the "develop" package of python and some build tools:

```shell
sudo apt-get install cmake ninja-build build-essential python3.11-dev
```

Optionally, you can also install `clang` as C compiler alternative, which improves the optimization of the binary
and performs slightly faster than GCC:

```shell
sudo apt-get install clang-18
```

Then, we need to install the CUDA toolkit and cuDNN packages from the Nvidia website. Follow the instructions provided
in the Nvidia website for [the CUDA toolkit](https://developer.nvidia.com/cuda-downloads) and [the cuDNN package](https://developer.nvidia.com/cudnn-downloads).

Before compiling, we also need to create a python virtual environment for OpenCV and activate it:

```shell
pwd # This will show the current path; we assume this returned '/path/to'
python3.11 -m venv opencv-venv
source /path/to/opencv-venv/bin/activate
```

The rest of this tutorial assumes that you have already activated the python virtual environment. Once all the
dependencies are installed, we proceed to clone the OpenCV python repository:

```shell
git clone --recursive https://github.com/opencv/opencv-python.git
```

We now proceed to configure the OpenCV package by setting CMake flags. *IMPORTANT:* Depending on the graphics card
you want to use, please configure the `-DCUDA_ARCH_BIN` argument _to match the compute capability_ of your graphics
card. To figure out which is the compute capability of your card, please [refer to the official Nvidia website](https://developer.nvidia.com/cuda-gpus).
The command shown below uses `clang` as the compiler for OpenCV. If you wish to use gcc, simply remove the `COMPILER`
lines in the cmake args:

```shell
export CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release \
                   -DWITH_CUDA=ON \
                   -DWITH_CUDNN=ON \
                   -DOPENCV_DNN_CUDA=ON \
                   -DENABLE_FAST_MATH=1 \
                   -DCUDA_FAST_MATH=1 \
                   -DCUDA_ARCH_BIN=7.5 \
                   -DWITH_CUBLAS=1 \
                   -DCMAKE_C_COMPILER=clang-18 \
                   -DCMAKE_CXX_COMPILER=clang++-18"
```

Additionally, we need to set up the OpenCV package to compile the "contrib" libraries, and (for headless environment)
create a headless package:

```shell
export ENABLE_CONTRIB=1
export ENABLE_HEADLESS=1
```

Then, start the compilation of OpenCV by simply creating the python wheel (Depending on your system, this might take
quite a while):

```shell
pip wheel . --verbose
```

Once compiled, you need to replace the installed OpenCV with the package just compiled _in the runtime virtual
environment_:

```shell
deactivate # Deactivate the OpenCV virtual env
source /path/to/virt-env/bin/activate # Activate the balrog runtime virtual env
pip uninstall opencv-contrib-python-headless
pip install opencv_contrib_python_headless-4.10.0.84-cp311-cp311-linux_x86_64.whl
```
