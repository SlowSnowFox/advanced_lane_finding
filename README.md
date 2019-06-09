

---

## Advanced Lane Finding Project


[//]: # (Image References)
[img1]: ./data/output_images/distortion_correction.jpg
[img2]: ./data/test_images/test1.jpg
[img3]: ./data/output_images/filter_sample.jpg
[img4]: ./data/output_images/persp_adj_sample_bin.jpg
[img5]: ./data/output_images/persp_adj_sample_or.jpg
[img6]: ./data/output_images/histogram_example.png
[img7]: ./data/output_images/Lane_Detector_sample.jpg
[img8]: ./data/output_images/Lane_tracer_sample.jpg
[img9]: ./data/output_images/complete_pipeline_sample.jpg
[img10]: ./data/output_images/distortion_correction_chessboard.jpg
[video1]: ./data/video_result.mp4 "Video"


---


### Camera Calibration

The camera matrix was calculated using the well know chess board reference.
A series of images of a chessboard was provided. Because all of the squares share a common z plane and we know the distances between the squares its relatively easy to compute the matrix by using cv2.calibrateCamera and cv2.findChessboardCorners.
The computed matrix is then saved and used as the first step in the pipeline.
The code for generating the matrix can be found under camera_calibration.
The code for actually applying the distortion correction can be found under
pipeline/filter_classes class `CamerAdjuster`
Below is an example of the step.
<br>

Original Image:
![Or Image][img2]

Distortion Corrected image:
![alt text][img1]

here is another example done on the chessboard:
![alt text][img10]

#### 2. Thresholding

I used a combination of color and gradient thresholds to generate a binary image.
Because the lane colors can vary under differnt light conditions I first converted them into HSL colorspace to make the filters more robust. In addition to this I also used a combination of gradient based filters. You can see the exakt composition I ended up using in pipeline/filter_classes class Lane_Detector.
Here is a combined image of the various filters and the resulting binary img.
![alt text][img3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.
I first experimented with different source and destination points before I ended up with the points that can be found under pipeline/find_lines.py.
The experimentation setup can be found under parameter_adjustment check_perspective_transform.py
I then used these fixed points to do the perspective transform in the pipeline(the class is PerspectiveAdjuster in pipeline/filter_classes)
Here are 2 example images. One used on the binary mask and the second one was used on the original image.
![alt text][img4]
![alt text][img5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

To identify the pixels that belong to the respective lanes I used 2 different approaches:

1. Without prior knowledge (Code can be found in the LaneSeparator class)
<br>
First I created a histogram along the x-axis which resulted in the following image:
![alt text][img6]

I then identified the 2 spikes and split the image in 6 slices along the y axes.
I placed the first 2 boxes on the centers of the 2 spikes. We get the n+1 box by moving up 1/6 of the image height and re-centering the box if the number of pixels within the new box surpasses a certain threshold(50 in our case).
Doing this until we cover the entire image results into an image like this:
![alt text][img7]

2. With prior knowledge (Code can be found in the trace_lanes function of the LaneTracer class)
<br>
Knowing where the last lane polynomial was we can simply  create a boundary around the old position and classify the pixels that are within the boundary as
belonging to the lane.
The resulting image looks like this:
![alt text][img8]


Once You have the pixels that belong to a line u can simply fit them with a second order polynomial. Why second order? We want to be able to follow curves but our function should not try to match the noise in the lane line pixels.
The fitting is done in the function pixel_to_lane in the LaneTracer class.
(You could see an example fit in the previous images)

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The curvature was calculated using the formula given [here](https://www.intmath.com/applications-differentiation/8-radius-curvature.php) . This was done in the Lane class in calculate_curvature.
The final curvature is the average of the left and the right lane.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The entire process over multiple frame is performed in the LaneTracer class.
To identify the road surface I simply fill the space between the 2 lane line polynomials.
Here is an example result on a test image:

![alt text][img9]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./data/video_result.mp4)

---

### Discussion

#### 1. Problems and room for improvement
There are still some minor problems in the current system:

  1. The current system only works well for highways.
  2. Extreme changes in light and/or a color change in the lane colors used could be problematic.
  3. Curvature calculation needs to be more precise
