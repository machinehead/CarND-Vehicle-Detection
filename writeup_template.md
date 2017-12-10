# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/hog_car.png
[image3]: ./examples/hog_non_car.png
[image4]: ./examples/search_region_1.png
[image5]: ./examples/search_region_2.png
[image6]: ./examples/pipeline_output_1.png
[image7]: ./examples/pipeline_output_2.png
[image8]: ./examples/pipeline_output_3.png
[video1]: ./project_video_ouput.mp4

---
### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The training pipeline can be found in the IPython notebook called TrainClassifier.ipynb.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of images contained in both classes:

![alt text][image1]

I then built a pipeline to explore different parameters for color binning, color histogram and HOG features (cell 4 of the notebook).

Eventually, I ended up not using color binning and color histogram, since HOG features alone worked good enough, and color-based features had a detrimental effect on detecting one of the cars in the project video.

Here is an example of HOG features using the `YUV` color space for a car and a non-car image:

![alt text][image2]

![alt text][image3]

#### 2. Explain how you settled on your final choice of HOG parameters.

I relied on RandomizedSearchCV in finding the parameters for HOG features (cell 4-8 of the TrainClassifier notebook).

The final choice of parameters was: `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(3, 3)`.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

First I started with LinearSVC, but I soon figured out that GradientBoostingClassifier outperforms LinearSVC in terms of both training speed and accuracy, so I switched to using GradientBoostingClassifier.
 
I tweaked the parameters for the classifier pipeline using code in cell 4 of the `TrainClassifier` notebook.

Finally, I've chosen the best performing set of parameters according to the cross-validation results and trained a classifier for that set of parameters (cell 4-8 of the TrainClassifier notebook).

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I chose a set of window sizes: `32x32`, `48x48`, `64x64`, `96x96`, `128x128`, `160x160`.
For each of those sizes, I visualized a grid over the entire image and highlighted positive classifier predictions in green.
After that, I've chosen a portion of the image where all the relevant matches were located, and changed the code to only run over that image portion.

Here are some of the results of that process:

![alt text][image4]

![alt text][image5]

Code and more image examples can be seen in the cell 2 of the Visuals notebook.
 
The implementation above slides through all the windows independently. The final implementation of sliding windows for video processing is different, since it uses HOG subsampling, and also resizes the search area so that the windows are `64x64` pixels, same as training set images. The optimized implementation can be found in the function called `find_bounding_boxes` in `tools.py`.  

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 5 window scales using YUV 3-channel HOG features.  Here are some example images (cell 4 in the `Visuals` notebook):

![alt text][image6]
![alt text][image7]
![alt text][image8]

The final pipeline used for processing the video is optimized by only evaluating HOG features once for the entire image and subsampling them later. Obtaining HOG features is a very computationally expensive step.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  

From the positive detections I created a heatmap. Every window contributes to the heatmap, but the weight contributed is inversely proportional to the window size, since I assume that bigger windows have less confidence about the actual location of the car. Every individual heatmap is thresholded to reduce the effect of bigger, single windows. 

After that I averaged the heatmaps across the last 6 frames and thresholded the average heatmap to reject spurious false positives.

I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.
  
When doing this, I preferred to err towards having less false positives. This leads to the bounding boxes being smaller than the cars, especially when seen from the side.

Images shown above display per-frame heatmaps and resulting bounding boxes, like this:

![alt text][image6]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The biggest issue I had with this project is that the training set seems to be sufficiently different from the recorded video, so that the white car in the project video fails to be detected by color-based features. Because of that reason, even when the classifier shows accuracy of over 99%, this number cannot be relied on.

I think that in order to make the classifier more robust, this project needs much bigger datasets (e.g. by incorporating the Udacity dataset, etc.). Otherwise different small changes that make test videos different from the training set will cause the pipeline to break.

Same as with another projects in this nanodegree, reviewing the result of the program manually takes time and doesn't guarantee measurable improvement. I think that in an actual enterprise setting, it would make sense to build some sort of automated evaluation mechanism, so that the quality of the entire video processing pipeline can be systematically evaluated.

