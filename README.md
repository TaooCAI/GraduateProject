# Deep Stereo Matching

Reproduce the work of Kendall A, et al. Meantime, I conduct some extra experiments for a further research.

## Requirements

* Python 3.6
* PyTorch 0.4.0

## Demos

**<p align="center">Left Image</p>**

<p align="center">
<img src="./demo/120/left.png?raw=true" alt="left image" height="60%" width="60%">
</p>

**<p align="center">Groundtruth Disparity</p>**

<p align="center">
<img align="center" alt="groundtruth" src="./demo/120/groundtruth.png?raw=true" height="60%" width="60%">
</p>

**<p align="center">Prediction Disparity</p>**

<p align="center">
<img align="center" alt="prediction" src="./demo/120/prediction.png?raw=true" height="60%" width="60%">
</p>

## Quantitative Results

Results on the Monkaa part of the SceneFlow datasets.

|Inputs Resolution|Outputs Resolution|Full resolution method|Average Endpoint Error (EPE)|
|:-:|:-:|:-:|:-:|
|540x960|540x960|2D Deconv + F + 0.5F + 0.25F|2.29|
|540x960|540x960|2D Deconv|2.45|
|540x960|540x960|Bilinear|5.69|

No more experiments was conducted in the whole SceneFlow datasets duo to some reasons.

## References

* Kendall A, Martirosyan H, Dasgupta S, et al. End-to-End Learning of Geometry and Context for Deep Stereo Regression[C]. international conference on computer vision, 2017: 66-75.
