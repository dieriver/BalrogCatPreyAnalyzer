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
As stated by the original repo, this software relies on tensorflow models that need to be downloaded separately.
Before doing this, we need to install some packages (in Debian-based systems):

```shell
sudo apt install libglu1-mesa-dev libglx-mesa0
```

## Protobuf compiler
The code also requires a rather old version of the protobuf compiler to work (v3.19.0). To install this version, please
go to the [asset page](https://github.com/protocolbuffers/protobuf/releases/tag/v3.19.0) of that release, and download
the build for your architecture.
For x86_65, you can simply run:

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

## Tensorflow models
To download the tensorflow models, we simply run (as stated in [tensorflow object detection repo](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-on-the-Raspberry-Pi)):

```shell
mkdir tensorflow1
cd tensorflow1
pwd # We use this to check the full path of the tensorflow models base folder (tensorflow1). We use it in the rest of this readme
git clone --depth 1 https://github.com/tensorflow/models.git
```

We now need to use the protobuf compiler to compile the tensorflow models:

```shell
cd /path/to/tensorflow1/models/research
protoc object_detection/protos/*.proto --python_out=.
```

We also need to download the SSD_Lite model from the [TensorFlow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
and unzip it in the tensorflow path:

```shell
cd /path/to/tensorflow1/models/research/object_detection
wget http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
tar -xzvf ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
```

## Python libraries and its python dependencies
Finally, we are ready to install this module and its dependencies. Please note that Balrog Cat Prey Analyzer depends
*on python 3.11*, and python 3.12 is not supported. To fulfill this, please run (in Debian-based systems):

```shell
sudo apt-get install python3.11-full
```

To install the runtime dependencies, it is recommended to create a python virtual environment (using the python3.11-venv
package which should have been installed with the command below):

```shell
python3.11 -m venv virt-env
```

Which will create a python virtual environment under the `virt-env` folder. To install this module and its dependencies,
we need to activate the virtual environment, and run pip to install this package:

```shell
$ source virt-env/bin/activate
(virt-env) $ pip install .
```


# Configuration
## Environment variables
Balrog uses a few environment variables to configure the interfaces: Camera input, Telegram Bot and Surepet login
details. To configure this, you need to execute the following lines in your shell:

```shell
export BALROG_TENSOFLOW_PATH=/path/to/tensorflow1/models/research
export CAMERA_STREAM_URI=<camera_rstp_url>
export SUREPET_USER=<surepet_user>
export SUREPET_PASSWORD=<surepet-password>
export TELEGRAM_BOT_TOKEN=<telegram_bot_token>
export TELEGRAM_CHAT_ID=<telegram_chat_id>
```

Please note that the `BALROG_TENSORFLOW_PATH` needs to contain an absolute path to the `models/research` folder of
the `tensorflow` package you unzipped before.
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

It is recommended to *not* modify the configurations under the `model` section, since they directly control the
sensitivity of the verdicts generated by the tensorflow model.

# Execution
To execute, simply activate your python virtual environment, export the required variables (if needed) and then simply
execute the module:

```shell
$ source virt-env/bin/activate
(virt-env) $ python3 -m balrog
```

This repository also contains a start script that you can use in a "production" environment:

```shell
$ source virt-env/bin/activate
(virt-env) $ ./balrog.sh
```

This script will start the module, but also restart the module if it fails for some reason. Additionally, you can use
the `balrog-dbg.sh` script to start the module in a similar manner, but using the `-m` option in python to get extra
debugging info from the python interpreter.
