# Transfer Learning 101

This is the entry point for our Intro to Transfer Learning workshop. The whole goal during this workshop is to go through the 
first steps for image classification with Neural Networks and implement the building blocks from scratch. 

If you want to do some reading beforehand, you'll find the following useful:

* [About Tensorflow](https://www.tensorflow.org/)
* [Different subsets of Tensorflow](https://www.tensorflow.org/guide/).
* Notions on image classification and re-training of classifiers in [the original version of this workshop](https://www.tensorflow.org/hub/tutorials/image_retraining#training_on_flowers).

## How do I get set up? ###

1. Make sure you have wget installed: `sudo apt-get install wget`.
2. Make sure you create a new virtual environment (instructions below).
3. Install the dependencies with the `requirements.txt` in this repo.
4. Notice for this workshop we will not need a GPU (all images are small and all trainings light).
5. There are two sides for this workshop:
    1. Using the current code as-is and understanding its effects (which will serve as an intro).
    2. Getting our hands dirty, re-implementing the code. We will start from branch `ẁorkshop_init`.
  
Python 3.5+ should work (tested with 3.5 & 3.6). Instructions to create a Python3 based environment are given below.

(If, instead of this, you are used to and prefer using docker, go right ahead! 
[Please choose one based on v1.12 from here.](https://hub.docker.com/r/tensorflow/tensorflow/)))


### TL;DR: Quick start with a virtual environment ###

Assuming you want to store your virtual environment under this same repo (FYI, other people prefer to do so under `$HOME/.venvs`):
```sh
~/TransferLearning$ python3 -m venv venv-tf
~/TransferLearning$ source venv-tf/bin/activate
~/TransferLearning$ pip install -r requirements.txt
```

#### GPU users

**Only** if you already have a GPU and **working CUDA** installation:

```sh
~/TransferLearning$ pip install tensorflow-gpu
``` 

### Pre-trained model & new dataset ###
**Linux users**: run this from the repo root folder:
```sh
~/TransferLearning$ bash sh/download.sh
```

**Other systems** (or if you want to do each step manually):
1. Download the pre-trained model from [here](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_1.0_224_frozen.tgz), into tf_files/models.
```sh
# Assuming you have wget installed (`sudo apt-get install wget`):
~/TransferLearning$ wget -P tf_files/models/ https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_1.0_224_frozen.tgz
```
2. Download and extract the sample data ([flowers dataset](http://download.tensorflow.org/example_images/flower_photos.tgz)) to tf_files.
```sh
# Assuming you have wget installed (`sudo apt-get install wget`):
~/TransferLearning$ wget -qO- http://download.tensorflow.org/example_images/flower_photos.tgz | tar --one-top-level=tf_files -xvz
```

#### Resulting files ####
Whichever method you chose, your folder structure should be:

```
~/TransferLearning$ tree tf_files/
tf_files/
├── flower_photos
│   ├── daisy
│   │    ├── image97999.jpg
│   │    └── ...
│   ├── dandelion
│   ├── roses
│   ├── sunflowers
│   └── tulips
├── models
│   ├── mobilenet_v1_1.0_224
│   │   ├── frozen_graph.pb
│   │   ├── labels.txt
│   │   └── quantized_graph.pb
│   ├── mobilenet_v1_1.0_224_frozen.tgz
│   └── save_mobilenet_tgz_here
...
```


## Finding your way ###
`sh` is your entry point; there's a bash script to showcase the usage of the retraining code.

`scripts` contains the retraining code; notice there have been some modifications to the original Tensorflow github script, 
but the interface is the same.

`tf_files` contains data, models and tensorboard logs (training_summaries), just
 like in the original Tensorflow Transfer Learning [tutorial](https://www.tensorflow.org/tutorials/image_retraining). However, you will be able to see
 stacked plots on the Tensorboard.


### Warning
Notice after you have played with the current state of the code, we will be re-implementing data loading, model loading 
and training.


## Tensorboard ####
Notice tf_files/training_summaries contains the following structure:
   * architecture
      * training_timestamp
        * train
        * validation

This way you will be able to compare the loss function and other metrics from different trainings in the same Tensoboard
plot. If you don't fully follow this currently, don't worry, it will be explained during the Workshop.

![tensorboard](doc/tensorboard_multiple.png)

Once you've already launched a training, go to a second terminal and
start your Tensorboard with:

```sh
~/TransferLearning$ tensorboard --logdir=tf_files/training_summaries &
```

Go with your browser to http://localhost:6006 and enjoy!


## Plan for the workshop

**[Check the wiki!](https://github.com/ividal/TransferLearning/wiki)**

Recap:

We'll be switching back and forth between the command line (to launch trainings) to
Tensorboard (to see what's going on).

We will be re-implementing different modules to load images, load pre-trained models and train.

The starting point being branch `workshop_init`.

## License ###
Modifications are under the same Apache license as the original Tensorflow code.
