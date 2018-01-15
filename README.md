# Transfer Learning 101

### How do I get set up? ###

1. Make sure you either:
  -- create a virtual environment
  -- or are running a docker container with docker ([see here](https//hub.docker.com/r/tensorflow/tensorflow/))
2. Install the dependencies with the `requirements.txt` in this repo.

Quick start:
```sh
$ virtualenv -p $(which python3) py-tf
$ source py-tf/bin/activate
$ pip install -r requirements.txt
```

Both Python 2.7 and Python 3.5 should work (tested with 2.7 and 3.5). 
... That said, it's time to let 2.7 go.

### License ###
I'm redistributing modications under the same Apache 2.0 license from the original Tensorflow code.