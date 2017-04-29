# Description

Here, we guide you step by step with a bare machine to get a real time object detector with Deep Learning Neural Network.

-  Claim:This project is based on [darknet](https://github.com/pjreddie/darknet)

-  Task: Real time object detection and classification

-  Paper: [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242)

-  Dependencies: Ubuntu 14.04/OpenCV v2.4.13/CUDA 8.0/cuDNN v5.0

-  Languages: C + Python 2.7 (Python 3.5 will work but you need to modify files in /scripts a little bit)

# Step by Step Tutorial

### Step-1. Prepare machine and environment

#### a. System and GPU

A `Ubuntu 14.04` native system is preferred in training process.
At least one `NVIDIA GPU Card` is required such as `GeForce` series to enable GPU mode. This is not a must but strongly recommended if you do not have lots of time.

*It is very tricky to use virtualBox on top of macOS/windows to communicate to GPU (actually, to be more general, taking advantages of host machine's resources including memories on VirtualBox is quite limited)*. *Therefore, I do not recommend this way but feel free to try it if you have no other work around.*

#### b. Environment (GPU)

##### Descriptions:

- [ ] `OpenCV` :  OpenCV is useful no matter whether you want to enable GPU mode and here we use `OpenCV v2.4.13` for `Ubuntu 14.04`;

- [ ] `NVIDIA Driver`: NVIDIA Driver is needed for machine to communicate with GPU;

- [ ] `CUDA`: CUDA is a parallel computing platform and application programming interface (API) model created by Nvidia and we use `CUDA 8.0` here;

- [ ] `cuDNN`: cuDNN is a GPU acceleration library, especially for deep learning neural networks (i. e., it speeds up when you work with GPU) and we use `cuDNN v5.0 for CUDA 8.0` here

##### Install:

`OpenCV`:

```
sudo apt-get update
sudo apt-get install -y build-essential
sudo apt-get install -y cmake
sudo apt-get install -y libgtk2.0-dev
sudo apt-get install -y pkg-config
sudo apt-get install -y python-numpy python-dev
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install -y libjpeg-dev libpng-dev libtiff-dev libjasper-dev
sudo apt-get -qq install libopencv-dev build-essential checkinstall cmake pkg-config yasm libjpeg-dev libjasper-dev libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev libxine-dev libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev libv4l-dev python-dev python-numpy libtbb-dev libqt4-dev libgtk2.0-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev x264 v4l-utils
wget http://downloads.sourceforge.net/project/opencvlibrary/opencv-unix/2.4.13/opencv-2.4.13.zip
unzip opencv-2.4.13.zip
cd opencv-2.4.13
mkdir release
cd release
cmake -G "Unix Makefiles" -D CMAKE_CXX_COMPILER=/usr/bin/g++ CMAKE_C_COMPILER=/usr/bin/gcc -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_V4L=ON -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON -D BUILD_EXAMPLES=ON -D WITH_QT=ON -D WITH_OPENGL=ON -D BUILD_FAT_JAVA_LIB=ON -D INSTALL_TO_MANGLED_PATHS=ON -D INSTALL_CREATE_DISTRIB=ON -D INSTALL_TESTS=ON -D ENABLE_FAST_MATH=ON -D WITH_IMAGEIO=ON -D BUILD_SHARED_LIBS=OFF -D WITH_GSTREAMER=ON -DBUILD_TIFF=ON ..
make all -j8
sudo make install
```

Referring from :https://gist.github.com/bigsnarfdude/7305c8d8335c7cfc91888485a33d9bd9

`NVIDIA Driver`:

```
sudo apt-get update && sudo apt-get upgrade
sudo apt-get install build-essential
wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run
chmod +x cuda_8.0.61_375.26_linux-run
mkdir nvidia_installers
./cuda_8.0.61_375.26_linux-run -extract=${PWD}/nvidia_installers
```

If you are using AWS EC2 Ubuntu Machines, also run following(which is to disable nouveau since it conflicts with NVIDIA's kernel module and please PAY ATTENTION TO THE LAST LINE, which requires REBOOT. Pay extra attention, if you are using AWS EC2 spot instance):

```
sudo apt-get install linux-image-extra-virtual
echo "blacklist nouveau" | sudo tee -a /etc/modprobe.d/blacklist-nouveau.conf
echo "blacklist lbm-nouveau" | sudo tee -a /etc/modprobe.d/blacklist-nouveau.conf
echo "options nouveau modeset=0" | sudo tee -a /etc/modprobe.d/blacklist-nouveau.conf
echo "alias nouveau off" | sudo tee -a /etc/modprobe.d/blacklist-nouveau.conf
echo "alias lbm-nouveau off" | sudo tee -a /etc/modprobe.d/blacklist-nouveau.conf
echo options nouveau modeset=0 | sudo tee -a /etc/modprobe.d/nouveau-kms.conf
sudo update-initramfs -u
sudo reboot
```

Then

```
sudo apt-get install linux-source
sudo apt-get install linux-headers-`uname -r`
sudo ./nvidia_installers/NVIDIA-Linux-x86_64-375.26.run -s
```

If you are not using AWS EC2 Ubuntu or, to be more clear, if you are using an Ubuntu with an UI, it means that you are running a X server, which will bring your problems on executing last line.

Therefore, you need to kill the X server to install nvidia driver (which by default will prompt out an UI) and then restart X server.

Therefore:

-[X] Press Control + ALT + F1

-[X] Type your ubuntu system username (exp. ubuntu) and password to log in

-[X] Kill X Server by `sudo service lightdm stop`

-[X] Navigate to the correct folder and Install NVIDIA Driver in silent mode `sudo ./nvidia_installers/NVIDIA-Linux-x86_64-375.26.run -s`

-[X] Restart X server by `sudo service lightdm start`

-[X] Go back to your fancy UI if you are using any by Control + ALT + F7


`CUDA`:

```
sudo modprobe nvidia
sudo apt-get install build-essential
sudo ./cuda-linux64-rel-8.0.61-21551265.run
sudo ./cuda-samples-linux-8.0.61-21551265.run
echo "export PATH=$PATH:/usr/local/cuda-8.0/bin" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=:/usr/local/cuda-8.0/lib64" >> ~/.bashrc
source ~/.bashrc
sudo ldconfig
```

`cuDNN`:

First,

download `cudnn-8.0-linux-x64-v5.0-ga.tgz` from https://developer.nvidia.com/cudnn (You may need to sign up a NVIDIA developer to download it and don't panic, it's free.)

![nvidia-cudnn](https://cloud.githubusercontent.com/assets/8921629/25516102/621cd374-2b9d-11e7-8afa-8351f700ced4.png)

Then:
```
tar -zxf cudnn-8.0-linux-x64-v5.0-ga.tgz
sudo cp ./cuda/lib64/* /usr/local/cuda/lib64/
sudo cp ./cuda/include/cudnn.h /usr/local/cuda/include/
```

Validate:

- [ ] `OpenCV`: ```pkg-config opencv --cflags```

- [ ] `NVIDIA Driver`: ```sudo nvidia-smi```

- [ ] `CUDA`: ``` nvcc --version```

- [ ] `cuDNN`: No need to validate, install cuDNN is purely just doing copy&paste to your local machine


### Step-2. Download this repo

```
git clone https://github.com/KleinYuan/easyYolo.git
```

And create a folder called `devkit` in root of this repo and also sub-folders like below:

```
+-- cfg
+-- scripts
+-- src
+-- devkit
|   +-- 2017
|       +-- Annotations
|       +-- ImageSets
|       +-- Images
|       +-- Labels
.gitignore
darkenet19_448.conv.23
easy.names
LICENSE
Makefile
README.md
```



### Step-3. Prepare data step by step

#### a. Collect Images with no bigger size than 416*416
Go ahead and collect many many images and put all of them into `devkit/2017/Images` folder, for example:
```
.
+-- dataSets
|   +-- 01.png
|   +-- 02.png
|   +-- 03.png
|   ..........
|   (many many many ...)
|   ..........
```


#### b. Install the correct tool
[labelImg](https://github.com/tzutalin/labelImg) is my favorite tool, which works very well on many different platforms and here we use this tool to label images:

#### c. Label images
`labelImg` will eventually create .xml file including bounding box coord, (class) name and file information

Sample .xml:
```
<annotation verified="no">
  <folder>2017</folder>
  <filename>0</filename>
  <path>${dir}/0.png</path>
  <source>
    <database>Unknown</database>
  </source>
  <size>
    <width>416</width>
    <height>416</height>
    <depth>3</depth>
  </size>
  <segmented>0</segmented>
  <object>
    <name>2</name>
    <pose>Unspecified</pose>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <bndbox>
      <xmin>239</xmin>
      <ymin>179</ymin>
      <xmax>291</xmax>
      <ymax>226</ymax>
    </bndbox>
  </object>
</annotation>
```

### Step-4. Pre-process labeled data and Configure model

#### a. Pre-process labeled data

First of all, you need to know how many classes you want to classify (i.e, how many kinds of objects you want this algorithm to spot eventually) and their names.

Then, open `scripts/easy_label.py` and edit the `7th line` to replace the `classes`. For example, the default one is for task to is train a model to spot `banana`, `monkey`, `panda` in future photos/videos of a zoo.

Also, in the same file, edit the `6th line` to replace the format of your images, and by default, it's `png`.

At last, add(replace the default) classes names in `./easy.names`.

Navigate to the root of this folder and run:
```
make prepare-data
```
Then you will see:
```
+-- devkit
|   +-- 2017
|       +-- Annotations
|           +-- 01.xml
|           +-- 02.xml
|           +-- ...
|       +-- ImageSets
|           +-- train.txt
|           +-- val.txt
|       +-- Images
|           +-- 01.png
|           +-- 02.png
|           +-- ...
|       +-- Labels
|           +-- 01.txt
|           +-- 02.txt
|           +-- ...
+-- 2017_train.txt
+-- 2017_val.txt
```
Which is exactly a DEV Kit for your deep learning model.

#### b. Configure Model

Firstly,

Let's say, your training images size is `A*A` (which means width and height are both A) and classes number (how many classes you are trying to classify) is B.

Navigate to the root of this folder and run:
```
python ${PWD}/scripts/in_place.py -f ${PWD}/cfg/easy.cfg -o ${IMAGE_WIDTH} -n A
python ${PWD}/scripts/in_place.py -f ${PWD}/cfg/easy.cfg -o ${IMAGE_HEIGHT} -n A
python ${PWD}/scripts/in_place.py -f ${PWD}/cfg/easy.cfg -o ${CLASS_NUM} -n B
python ${PWD}/scripts/in_place.py -f ${PWD}/cfg/easy.data -o ${CLASS_NUM} -n B
```

Secondly,

We need to do some math and let's say you have C = (classes + 5) * 5.

Then continue run:

```
python ${PWD}/scripts/in_place.py -f ${PWD}/cfg/easy.cfg -o ${FILTERS_NUM} -n C

```

### Step-5. Make darknet executable and Train

#### a. Configure Makefile

If you are using GPU to train, then do not change any thing.

If you are using CPU to train, open Makefile and edit first 4 lines to be below:
```
GPU=0
CUDNN=0
OPENCV=1
DEBUG=0
```
which basically just disable GPU mode



#### b. Create darknet executable

Navigate to the root of this folder and run:
```
make
```
Then you are supposed to see a file just called `darknet` in root.

#### c. Train

Navigate to the root of this folder and run:
```
make train
```


*Command above will train model with single GPU*

If you want to train with multiple GPUs you need to first still run above and then wait for a model called easy_1000.weights (or whatever weights files larger than 1000) occur in backup folder.

And then run:
```
make train-multi-gpus
```
(Examples above assume that you have 2 GPUs and with easy_1000.weights)


TADA! You can go to sleep and wait for several hours (days) to get a trained model sitting in `backup` folder.

### Step-6. Test with static image and live stream camera

#### a. Static Image
Drag your image into the root folder and rename it as `test.png`, then run:
```
make test
```

#### b. Live stream camera

```
make test-camera
```

# Further Work

- [X] Add a python wrapper
- [ ] Build a docker image to automate the entire process (could be tricky to mount with camera tho)

# License
[License](LICENSE)