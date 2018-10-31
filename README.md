# Transfer Learning 101

## How do I get set up? ###

1. Make sure you either: 
    * create a virtual environment 
    * or are running a docker container with docker ([images available here](https://hub.docker.com/r/tensorflow/tensorflow/))
2. Install the dependencies with the `requirements.txt` in this repo.

Python 3.5+ should work (tested with 3.5 & 3.6). Instructions to create a Python3 based environment are given below.

Conda (Anaconda) is _known_ to cause all sorts of problems and does way too much magic for anyone to investigate when 
things go badly. It is strongly encouraged to use venv (already available with your Python installation) instead.

(If, instead of this, you are used to and prefer using docker, go right ahead!)


### TL;DR: Quick start with a virtual environment ###

Assuming you want to store your virtual environment under this same repo (FYI, other people prefer to do so under `$HOME/.venvs`):
```sh
$ python3 -m venv py-tf 
$ source py-tf/bin/activate
$ pip install -r requirements.txt
```

### Pre-trained model & new dataset ###

1. Download the pre-trained model from [here](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_1.0_224_frozen.tgz), into tf_files/models.
```sh
# Assuming you have wget installed (`sudo apt-get install wget`):
wget -P tf_files/models/ https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_1.0_224_frozen.tgz
```
2. Download and extract the sample data ([flowers dataset](http://download.tensorflow.org/example_images/flower_photos.tgz)) to tf_files.
```sh
# Assuming you have wget installed (`sudo apt-get install wget`):
wget http://download.tensorflow.org/example_images/flower_photos.tgz | tar --one-top-level=tf_files -xvf
```
4. Your folder structure should be:

```
(py-tf) user@machine:~/TransferLearning$ tree tf_files/
tf_files/
├── flower_photos
│   ├── daisy
│   │    ├── image97999.jpg
│	│	 └── ...
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
`sh` is your entry point; there's a bash script to showcase the usage of the retraining code

`scripts` contains the retraining code; notice there have been some modifications to the original Tensorflow github script, 
but the interface is the same.

`tf_files` contains data, models and tensorboard logs (training_summaries), just
 like in the original Tensorflow Transfer Learning [tutorial](https://www.tensorflow.org/tutorials/image_retraining). However, you will be able to see
 stacked plots on the Tensorboard.


## Tensorboard ####
Notice tf_files/training_summaries contains the following structure:
   * architecture
      * training_timestamp
        * train
        * validation

This way you will be able to compare the loss function and other metrics from different trainings in the same Tensoboard
plot.

![tensorboard](doc/tensorboard_multiple.png)

Once you've already launched a training, go to a second terminal and
start your Tensorboard with:

```sh
tensorboard --logdir=tf_files/training_summaries &
```

Go with your browser to http://localhost:6006 and enjoy!


## Plan for the workshop

**[Check the wiki!](https://github.com/ividal/TransferLearning/wiki)**

Recap:

We'll be switching back and forth between the command line (to launch trainings) to
Tensorboard (to see what's going on).

* Tweak parameters inside sh/retrain.sh and launch with
```
sh/retrain.sh
```

* Launch Tensoboard only once with 
```
tensorboard --logdir=tf_files/training_summaries
```


## License ###
Modifications are under the same Apache license as the original Tensorflow code.
