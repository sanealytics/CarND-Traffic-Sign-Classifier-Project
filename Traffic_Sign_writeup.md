#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./class_distribution_training.png "Training Class distribution"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

[RightOfWay]: ./newdata/100_1607.jpg "Right of way at next intersection"
[20]: ./newdata/20.jpg "20 km/h"
[30]: ./newdata/canstock14957677.jpg "30 km/h"
[chidcross]: ./newdata/german-traffic-signs-picture-id459381295.jpg "Children crossing"
[pedestrian]: ./newdata/german-traffic-signs-picture-id469763303.jpg "Pedestrian crossing"
[bicycles]: ./newdata/germany-neuharlingersiel-no-bicycles-sign-on-beach-emc7k6.jpg "Bicycles"
[30_2]: ./newdata/stock-photo-german-traffic-sign-indicating-a-zone-with-reduced-traffic-and-a-speed-limit-of-kilometers-per-249213382.jpg "30 kmph"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. 

This report will cover each rubric point. Here is the link to my [project code](https://github.com/sanealytics/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32 x 3 channels
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the fourth code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing the distribution of examples by class in the training data.

![alt text][image1]

The notebook compares this to train and validation sets as well. It's mostly balanced.

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the 7-9 code cells of the IPython notebook.

As a first step I shuffled the training data.

I then added a few transforms to the model itself. First, I normalized. It gave modest gains but not any after the next transform, so I removed it from the final model.

I included a grayscale transform. This gave a surprisingly nice lift. I included this because intuitively colors didn't seem to add much as I looked through some traffic signs. And it gives a very modest reduction in parameters.

I then defined random_image_transforms() that will, when in training mode, randomly do various image transformations - flip_left_right, random_brightness, random_contrast, random_saturation, and will sometimes leave the image alone. This was to increase the size of the dataset while training by augmenting with all these transformed images.

This is also very fast as it will do this at train time without having to pre-generate augmented data. And speed is a nice cost effective advantage on AWS.

It seemed to have very minor effect after grayscale, so I left it in there but unfortunately, not as large as I had hoped. It is possible there is some error in the code but I could not find it.


####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

I described this in the above section already. I included these into the model itself. I had to add a phase variable that will make the random transforms take effect only in training phase. This is put together in Cell 13.

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the tenth cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x6   	|
| RELU					|												|
| Dropout       |                       |
| Avg pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 10x10x16   	|
| RELU					|												|
| Dropout       |                       |
| Avg pooling	      	| 2x2 stride,  outputs 5x5x16   				|
| Fully connected		|  output 84        									|
| RELU					|												|
| Fully connected		|  output 43        									|
| Softmax       |												|


I tried batchnorm but unfortunately, it did not better the validation error. I suspect I don't know how to use it properly. I would invest more time in that and also adding 1x1 convolutions if I had more time.

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 13-15 cells of the ipython notebook. 

To train the model, I used a AdamOptimizer with a small learning rate (0.0001) and 100 epochs. I had run a lot of experiments with different configurations.

I also saved the best model I had seen so far instead of the last one.

All the experiments are described as html in Markup in cell after 15.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 17th cell of the Ipython notebook.

My final model results were:
* training set accuracy of 0.996
* validation set accuracy of 0.949
* test set accuracy of 0.929

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

I started with LeCun architecture that was given as part of the lab. It made sense to start with an architecture I understood well from class.

* What were some problems with the initial architecture?

It worked well enough actually. I only made minor changes to it. Most changes did not bring a big change.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

I ran the following experiments:
```
    Base:
        lr=0.01, 3 channels, epochs=20
        EPOCH 20 ...
        Accuracy Training = 0.995 Validation = 0.897

    Not a bad start

    Normalize:
        lr=0.01, 3 channels, epochs=20, scaled
        EPOCH 20 ...
        Accuracy Training = 0.938 Validation = 0.783

    Became worse... don't normalize
      
    Grayscale:
        lr=0.01, 1 channels, epochs=20
        EPOCH 20 ...
        Accuracy Training = 0.997 Validation = 0.942

    Really nice improvement over base

    Random:
        lr=0.01, 3 channels, epochs=20
        EPOCH 20 ...
        Accuracy Training = 0.990 Validation = 0.905

    Nice improvement over base
    
    Random + Grayscale:
        lr=0.01, 1 channels, epochs=20
        EPOCH 20 ...
        Accuracy Training = 0.997 Validation = 0.942

    Together, not that much better, overfits surprisingly

    Random + Batchnorm:
        lr=0.01, 3 channels, epochs=30
        EPOCH 30 ...
        Accuracy Training = 0.974 Validation = 0.902
    Overfits. Also, seems to oscillate a lot, maybe lower learning rate

    Random + Grayscale + Batchnorm:
        lr=0.0001, 1 channels, epochs=30
        EPOCH 30 ...
        Accuracy Training = 0.910 Validation = 0.820

    Didn't go as planned. Maybe need to train longer or some error in code, but don't have time to experiment
        
    Random + Grayscale + Dropout + Keep best model:
        lr=0.0001, 1 channels, epochs=100
        EPOCH 100 ...
        Accuracy Training = 0.996 Validation = 0.949
    Final solution
```

* Which parameters were tuned? How were they adjusted and why?
Dropout was added because we were overfitting after data augmentation.

The learning rate was adjusted after dropout because curves looked jumpy.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

Convolution kernel is the ideal solution to this problem as it works well for images. Dropout was added after overfitting was confirmed looking at loss curves.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are seven German traffic signs that I found on the web:

![alt text][RightOfWay] ![alt text][20] ![alt text][30] 
![alt text][ChildCross] ![alt text][Pedestrian]
![alt text][Bicycles] ![alt text][30_2]

It actually did horribly on most of these, sans one. I had thought that the image transforms will get us far. But these images did not work because they are too different from the training data.
Training data images are nicely centered whereas these images are all over the place. 


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 15th cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed Limit (30km/h)| Speed Limit (60km/h)   									| 
| Bicycle crossing     			| Road narrows to the right										|
| Speed Limit (20km/h)| Speed Limit (60km/h)   									| 
| Right-of-way at the next intersection | Right-of-way at the next intersection |
| Speed limit (30km/h) | Road Work |
| Children crossing | General caution |
| Pedestrians | No passing |


The model was able to correctly guess 1 of the 7 traffic signs, which gives an accuracy of 14%. This compares very unfavorably to the accuracy on the test set of 93%

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 26th cell of the Ipython notebook.

In general it got that certains signs were about speed limit, or were round. But it did not get the exact speed limit.

The main problem is that there are things behind the signs. It got the sign where there was only sky behind it. 

There are a few ways to takle this problem: 

* Generate dataset with different backgrounds to help the model learn to pick the sign from the noise
* In real life, we might have the advantage of depth and time. So as the camera moves with the car, we might be able to better isolate what is in front vs back.

