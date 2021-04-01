[//]: # (Image References)

[model_architecture]: ./images/model.png "Model"
[training_loss]: ./images/training_loss.png "Training Loss"

---
# Behavioral Cloning Project

Overview
---

<video src="run1.mp4" width="320" height="200" controls preload></video>

This repository contains starting files for the Behavioral Cloning Project.
In this project, I've used what I've learned about deep neural networks and convoltional neural networks. 
Using this learning, a driver's driving behavior is cloned to drive the car in simulator autonomously.
<br>I've trained, validated and then tested the model with self generated training and validation set using Udacity SDC's provided simulator. The model was trained to predict the steering angle for autonomous car.

To meet specifications from [rubric points](https://review.udacity.com/#!/rubrics/432/view), the project has below files: 
* model.py (script used to create model)
* main.py (script to traverse through training data and create it for feeding into the network. Also contains training part.)
* drive.py (script to drive the car. Modified for increasing thorttle)
* model.h5 (a trained Keras model)
* README.md (markdown file for explanation)
* [run1_full_track1.mp4](https://www.youtube.com/watch?v=d9KjyGlr_iQ) (a video recording of vehicle driving autonomously around the track for one full lap)

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies
This requires:
* Keras: For model creation and training
* matplotlib.pyplot: Plotting the loss we get so that we understand how well model is doing
* openCv: For reading images
* numpy: For array operations
* drive.py: For driving the car taking saved model as an input
* video.py: For outputting the video of saved images from autonomous driving
* Simulator from udacity: For collecting training data and inferencing

## Details About Files In This Directory

### `model.py`
This is a file which defines the model that we have used in the project. It contains a function named `nvidiaModel()` which returns the [model](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) developed by nvidia for their self driving cars, with just a single change.
<br> That one change is removal of a `fully connected layer` of 1164 neurons.<br>
![model_architecture]
* My model looks like this.
* This image is generated by [Keras' Visualization](https://keras.io/visualization/) APIs
* For using this model, one just needs to import `nvidiaModel()` function from this file.

```python
from model import nvidiaModel
model = nvidiaModel()
```
### `main.py`

This file is responsible for creating a data suitable to feed in to the network and train the network.
<br>
It is **assumed** here that, the training data is present to the relative path at `../behavioral_cloning_data/Track1/<multiple folder containing IMG and driving_log.csv>/`. 
<br>The data is present in different directories here which is for:
* Whole track1 in the default direction
* Whole track1 in opposite direction so that car thinks it's totally a different track
* Dirt road

A method named `loadData()` is used to traverse through the available directories and create an augumented data so that our training samples increase in count.
<br>
While collecting the image data, we're **flipping** the image virtically so that the number of sample count will double automatically.
But here, we also need to keep in mind that, when we flip the image virtically to get the right turn converted into the left, we also need to change its steering angle to exactly opposite.<br>
Ex. For right turn, our steering angle was suppose `0.67` degrees, thus after flipping the image, the right turn became left and steering angle becomes `-0.67` degrees.
<br>
Thus, while augumenting the images, had to multiply the flipped images' steering angle by `-1`.
<br>
The file works in below steps:
1. List all the directories under provided path
2. Call `loadData()` with list of directories as an argument
3. Create `X_train` and `y_train` where `X_train` is a training data and `y_train` is a label data; in our case, it's a steering angle.
4. Call `nvidiaModel()` from `model.py` to get the nvidia model.
5. Compile the model and train for desired number of epochs. (I did it for 30 epochs).
6. Record the losses we obtained for every epoch and plot the graph to understand how well the model is doing.
<br>
My model did pretty well and thus got below `loss vs epochs` graph.
<br>

![training_loss]

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. Below code depicts how to save the model.

```python
model.save('model.h5')
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```
The file is slightly modified to change the throttle value to 12.
<br>
The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1_full_track1
```

The fourth argument, `run1_full_track1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1_full_track1

2018_01_10_21_10_23_424.jpg
2018_01_10_21_10_23_451.jpg
2018_01_10_21_10_23_477.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1_full_track1
```

Creates a video based on images found in the `run1_full_track1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1_full_track1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1_full_track1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.
