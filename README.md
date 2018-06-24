##Writeup - Vehicle Detection and Tracking Project

---

**Vehicle Detection Project**

![alt text][image1000]

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./DocImages/Car_NoCar_Images.png
[image2]: ./DocImages/HOG_Channel0.png
[image3]: ./DocImages/SVM_Train_Results.PNG
[image4]: ./DocImages/SVM_Train_Results2.PNG
[image5]: ./output_images/test1_Final.png
[image6]: ./output_images/test5_Final.png
[image7]: ./output_images/test1_boxes.png
[image9]: ./output_images/test1_heatmap.png
[image10]: ./output_images/test1_filtered.png
[image8]: ./output_images/test1_labelled.png
[image11]: ./output_images/test1_Final.png

[image001]: ./output_images/test1_Combined.png
[image002]: ./output_images/test2_Combined.png
[image003]: ./output_images/test3_Combined.png
[image004]: ./output_images/test4_Combined.png
[image005]: ./output_images/test5_Combined.png
[image006]: ./output_images/test6_Combined.png

[image1000]: ./output_images/ResultOutput.gif

[image101]: ./VideoCaptureOutput/frame1_Combined.png
[image102]: ./VideoCaptureOutput/frame2_Combined.png
[image103]: ./VideoCaptureOutput/frame3_Combined.png
[image104]: ./VideoCaptureOutput/frame4_Combined.png
[image105]: ./VideoCaptureOutput/frame5_Combined.png
[image106]: ./VideoCaptureOutput/frame6_Combined.png
[image107]: ./VideoCaptureOutput/frame7_Combined.png
[image108]: ./VideoCaptureOutput/frame8_Combined.png
[image109]: ./VideoCaptureOutput/frame9_Combined.png
[image110]: ./VideoCaptureOutput/frame10_Combined.png

[image201]: ./VideoCaptureOutput/MultiFrame1_Combined_TH25.png
[image202]: ./VideoCaptureOutput/MultiFrame2_Combined_TH25.png
[image203]: ./VideoCaptureOutput/MultiFrame3_Combined_TH25.png
[image204]: ./VideoCaptureOutput/MultiFrame4_Combined_TH25.png
[image205]: ./VideoCaptureOutput/MultiFrame5_Combined_TH25.png
[image206]: ./VideoCaptureOutput/MultiFrame6_Combined_TH25.png
[image207]: ./VideoCaptureOutput/MultiFrame7_Combined_TH25.png
[image208]: ./VideoCaptureOutput/MultiFrame8_Combined_TH25.png
[image209]: ./VideoCaptureOutput/MultiFrame9_Combined_TH25.png
[image210]: ./VideoCaptureOutput/MultiFrame10_Combined_TH25.png


[video1]: ./test_video_out.mp4
[video2]: ./project_video_out.mp4
[video2]: ./project_video_out_multiFrame.mp4

## [Rubric Points](https://review.udacity.com/#!/rubrics/513/view)
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  


###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the the function 'get_hog_feature' in the Functions.py (Line# 19 to Line #36), and also used in the function extract_features and find_cars in the Functions.py.  

I started by reading in all the `vehicle` and `non-vehicle` images (Line #21 to Line # 38 in Classifier.py).  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`, and I only used Channel 0 for the HOG parameter extracts:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters including Color space the number of the orientation (orient), pix_per_cell, cell_per_block, and the channel used for hog (hog_channel). Start with the low number, like 6 orientation, 8 pixel_per_cell, and 2 cell_per_block. And I found most of the setting I tried have OK testing score, above 0.95. And finally, I settled on 'YUV' color space, orient = 9, pix_per_cell = 8, cell_per_block =2, and hog_channel = 0. Also, I have color histogram (hist_bins = 32), and spatial vector (spatial size = (32,32)) and I got more than 0.97 testing score. The total feature vector length is 4932 (which I think is little too long.) 

![alt text][image3]

Initially, I didn't know it will take long time to process the video (more than 3 hours on the project video), using the feature vector legnth with such long. Once I found the processing time is too long, I decided to reduce the spatial size to (16,16) and color bins to 8, and increase oritentation number to 11. Then, I got feature vector length of 2948, and the accuracy of SVC increases to 0.98. 

![alt text][image4]



####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using provided cars and non-cars images in size of 64x64 pixels. First of all, I extract the feature vectors (also calculate the StandardScaler for the feature vector's scaling using all data set) from all cars and non-cars images, and then, I divided the vector data set (8792 car images, and 8968 non-car images) into 80% for the training and 20% for validation data set along with corresponding y-value with random sequence.

In the second step, I used the training data to train the SVM model, and use the validation data to measure the efficiency of the training. As shwon above, with Spatial, Color Histogram, and HOG of 2948 vectors in total, the trained SVM could have accuracy score of 0.98.

The data feature extraction, scaling, SVM model training and validation could be found from line # 119 to line # 216.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window is integrated in the pipeline to find the car using the extracted features. Here I just explain the sliding window search part. I will describe the classifier in the next section.

The sliding window function could be find in find_cars() function from line #262 to # Line # 375 in Function.py. Also, the sliding window with multiple scaling on the image could be found in from Line # 223 to Line # 234 & from Line # 279 to Line # 288 in Classifier.py.

As an image read into the find_cars function, I will scale the image to [0,1], then, cut out the part of image where other cars could be shown up, which is lower half part of image [400 - 600] in y-axis. Then, I converted the image into 'YUV' color space, which is the same as the parameters used in the SVM training. In the next step, I scale the image in order to make the classifier work on small or larger car object. Then, the number of blocks are defined based on HOG's parameters, and the HOG features extraction would naturally divide the image into multiple blocks and windows with different sizes, in the size of cell, block, and window. From each window, if the window size is not 64x64, I need to resize the windowed image to 64x64 since the classifier works on 64x64 image. Also, the spatial features and color histogram features are extracted from the windowed images. The combined features from HOG, color and spatial are sent to trained SVM for car detections. If the prediction is positive, the window's info will be saved in the drawing box list for the further process, including: heatmap, filtering, labeling, and final detection display.

Also, I would run the car scanning with different scales (from 0.5 to 2 with step of 0.3), in order to capture the car in the different size. And all positive windows would be used for heatmap and thresholding filter.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on multiple scales from 0.5 to 2 with step of 0.3 using YUV channel-0 (Y) HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image5]

Let's check the pipeline step by step:

(1) The classifier would return multiple windows from image on the different scales:

![alt text][image7]

(2) In order to make the detection more robust, the detector needs to filter out the false detections by checking on the reliablity of the detetions based on the counts of dections at the same location in different window size and scales. Therefore, the heatmap is generated showing how reliable of the detection is:

![alt text][image9]

(3) Apply a count number filter, and use the label function to make the detection into clusters, in this way, some random false detection could be removed:

Heatmap after filter:

![alt text][image10]

Labels:

![alt text][image8]

(4) The final image showing the detection is by drawing the label area with the rect:

![alt text][image11]

The parameters are optimized based on the issues shown in the detection images: (1) Adjust the feature extracts to reduce the number of false detections; (2) Adjust the number of scalings to make the scaling matched with the image/video's vehicle size; (3) Adjust the threshold to remove the false detections. The pipline code could be find from Line # 218 to #271 in Classifier.py.

The follwoing ar the result on the test images:

![alt text][image001]

![alt text][image002]

![alt text][image003]

![alt text][image004]

![alt text][image005]

![alt text][image006]


---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a video using multiple scales on each frame. ![link to my video result][video2]

The video using multiframe function is as following:
Here's a video using multiple frames for heatmap and labeling, every four frames as a set, and frame in each set is applied different scale. ![link to my video result][video3]
I also tried with different thresholds, in this video I set to threshold to very high value (50).

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

As shown in the previous section, I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Also, since the video has the continous frames and the car moves in the adjacent pixels. Therefore, the labeled area in the previous frame could be filtered by lower threshold since it is more reliable if the car detection happened within the similar area in this frame. By doing this, we are able to set higher threshold on other areas, which could be helpful on immunity to more false detections. The code of the dynamic threshold is in the function apply_threshold() from Line # 389 to Line #401 in Functions.py. Also, due to high similarity of adjacent frames, it is able to using multiple frame as a frame set, and applying different scaling factor to different frames, and use the combined detection windows for the heatmap filtering and labeling.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video. Please note that multiple scaling factors are applied to each frame.

### Here are ten frames and their corresponding heatmaps, filtered heatmap after threshold, output of `scipy.ndimage.measurements.label()` on the integrated heatmap, and the resulting bounding boxes are drawn onto the frame in the series, the previous frame's label area has impact on the next frame by using the lower (2/3 of the standard threshold) on the previous detected area:

![alt text][image101]

![alt text][image102]

![alt text][image103]

![alt text][image104]

![alt text][image105]

![alt text][image106]

![alt text][image107]

![alt text][image108]

![alt text][image109]

![alt text][image110]

From the previous images, it could be seen that the images are really similar to each other around the adjacent frames. Therefore, I could use multiple frames for more reliable and faster detection. In the previous video_process() function, multiple scaling numbers are applied to one frame, which make the process really slow. In the updated function video_process_MultiFrame(), I applied four different scaling number on four adjacent frames, and use the positive detection windows in last four frames to generate heat map and filter the false detections. The results showed that the window detection and tracking is more smooth and stable.

![alt text][image201]

![alt text][image202]

![alt text][image203]

![alt text][image204]

![alt text][image205]

![alt text][image206]

![alt text][image207]

![alt text][image208]

![alt text][image209]

![alt text][image210]

The multiple-frame pipeline implementation is from Line#387 to Line #430 in function video_process_MultiFrame() in Classifier.py

[![Vehicle Detection](http://img.youtube.com/vi/XE-vP--zZfQ/0.jpg)](http://www.youtube.com/watch?v=XE-vP--zZfQ "Vehicle Detection")

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
1. One challenge I would like to address in future is to detect the vehicle far away from the point of view. Due to the long distance, the size of the vehicle is too small to be detected by the classifier. One possible solution is to use the perspective transfer as we used in the previous project, in order to recover the size of the vehicle. Another advantage is the perspective transfer could make reduce the scaling numbers.
2. There is still room to improvement on the accuracy of the classifier, especially to reduce the false detections. Possible method is to use kernel for the SVM, multiple classifier to detect different objects, take the information from the lane detections to better estimate the possible location of the vehicles.
3. The frame rate or processing time could be reduced further in order to the real-time applications. Other than the hardware upgrade and using GPU/ASIC acceleration, one possible method is to use different feature vectors at different frames, like the current method using different scales at different frame, and then to generate the heatmap based on the multiple frame's decision. In this way, it is possible to reduce the average feature vector size by half or more, since it is possible just applying one high accuracy detection every 10 frames or more, and other frames just use half or 1/3 vector features for the detections.
