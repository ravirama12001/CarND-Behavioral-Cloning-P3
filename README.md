# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Step by step guide:
---
I will give you Step by step guild on how I have achieved this project.

**Step1**: 
	Get the Udacity base project for CarND-Behavioral-Cloning-P3

**Step2**:

	Generate Data using Car Simulator. 
	Remember more data means more accuracy.
	I have recorded 3 laps for good accuracy.

**Step3**:

	Now take all images as X_train for model and remaining data as Y_train and input to model.
	Learn more how to create keras models.


**Step4**:

	Once Data is Collected now good to Create model using the gathered data.

I have used nvidia [model](https://devblogs.nvidia.com/deep-learning-self-driving-cars/)

Modify **model.py** to create model. You need to save file to model.h5

Once model.py completed, run below

python model.py to create model.h5 as in my code

**Step5**: 

	**drive.py** is the file to verify the model

	Use below command to verify autonomous

> 		python drive.py model.h5 run1

	This will enable you to run the sumulation atonomously using the model created and also saved all image files created under this run into run1 foloder.

**Step6**:

	Now you can create a video out of atonoumous run using **video.py** 

> 		python video.py run1

		This will create run1.mp4 file
