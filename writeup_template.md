##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/vehicle.png
[image2]: ./output_images/novehicle.png
[image3]: ./output_images/test1_pipeline.png
[image4]: ./output_images/vehicle(7X6X2).png
[image5]: ./output_images/vehicle(8X8X2).png
[image6]: ./output_images/vehicle(9X8X2).png
[video1]: ./out_test_video.mp4
[video2]: ./out_YUV_test_video.mp4

###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  


###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (p5.ipynb).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt text][image2]



####2. Explain how you settled on your final choice of HOG parameters.

I use the parameters as follow:

colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)
hist_bins = 32
spatial_feat=True
hist_feat=True
hog_feat=True

I tried various combinations of parameters and record the result of accuracy of svm training. the record file is ./hog_svm_parameter.text, when I don't include spatial or hist feature, the accuracy is below 0.99.  if I only include hog of one channel feature, the accuracy is also below 0.99. 

when I select 'YUV' space, the accuracy is 0.993, although 'YCrCb' is only 0.9907. but the YUV vidio output has more false positions (./out_YUV_test_video.mp4)

I test orient\pix_per_cell\cell_per_block to test the hog feature, the visual images:
![alt text][image4]
![alt text][image5]
![alt text][image6]
so I select orient = 9 pix_per_cell = 8 cell_per_block = 2, I think it find feature enough to classify.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the follow code , it is simple and no any trick. all the trick is in  preprocess data, such as StandardScaler and Split up data into randomized training and test sets.
svc = LinearSVC()
svc.fit(X_train, y_train)

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I use the code of " find_cars " in lesson,  it  only has to extract hog features once and then can be sub-sampled to get all of its overlaying windows.

I use the follow parameter:
scale = 1.5, because we use window=64, but the window is small for out vidio detect car. so we need to expand our window. and we use scale =1.5 to resize the input image size, so the window we indeed use is about 96.

we use cells_per_step = 2  # Instead of overlap, define how many cells to step, about we use pix_per_cell = 8 , so our overlap is 0.75.  
 we use scale = 1.5 and cells_per_step = 2 to balance the performance and the accuracy of vehicle detection

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

![alt text][image3]  and also all six images display in P5.ipynb file
I optimize the performance use the follow parameters to decrease the search windows.
ystart = 400
ystop = 656
scale = 1.5

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)



Here's a [link to my video result](./out_test_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  
I also integrate a heat map over 30 frames of video, such that areas of multiple detections get "hot", while transient false positives stay "cool". I  can then simply threshold (15)  my  heatmap to remove false positives.


---

### Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

1  firstly I think the performace is not good, my pipeline can only process 4 frame per second,   I think I can find how to use hog algorithm in GPU in future to test the performance.
2  and I test object detection algorithm of deep learning, Yolo, It can achive about 20~30 frames per second. and its accurate is good too. so I prefer deep learning algorithm in future.


