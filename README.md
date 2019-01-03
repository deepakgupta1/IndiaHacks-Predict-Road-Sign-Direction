# IndiaHacks-Predict-Road-Sign-Direction
Predict direction of road sign facing from data collected from vehicles of Here Technologies.

### Problem
Building digital maps is challenging, and maintaining it up to date in an ever-changing world is even more challenging. Various machine learning techniques helps us to detect road signs and changes in real world, and process it to update maps.

The problem presented here is related to a step after detecting a sign on a road. This step has to now identify each road geometry on which this sign is applicable. While sounds like a simple problem, signs in junctions makes this more challenging.

For example, given a sign detected on a road from a 4-camera setting on vehicle, the closest sighting of the sign may be in the right facing camera, with a sharp sign angle with respect to the direction of the car on which cameras set is mounted. Next step for updating map using this sign is to identify the exact road on which this sign is to be placed or applied.

### Data
Dataset provided here has details on camera sign was detected, Angle of sign with respect to front in degrees, Sign's reported bounding box aspect ratio (width/height), Sign Width and Height, and the target feature Sign Facing, which is where the sign is actually facing.

### Approach:
* OHE of DetectedCamera
* Made some features
* 5 fold cv.
* xgbclassifier.

### Features:
* Quadrant in which detected.
* How far from the axis the angle is like if angle is 105, then (105-90) = 15. So there are four axis- at 0, 90, 180 and 270.

### Major Libraries:
* sklearn
* xgboost
* pandas
* numpy
* datetime
