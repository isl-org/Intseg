# Interactive Image Segmentation with Latent Diversity
This is a Tensorflow implementation of Interactive Image Segmentation with Latent Diversity. It receives positive and negative clicks and produces segmentation masks.

## Setup

### Requirement
Required python libraries: Tensorflow (>=1.3) + OpenCV + Scipy + Numpy.

Tested in Ubuntu 16.04 LTS + Intel i7 CPU + Nvidia Titan X (Pascal) with Cuda (>=8.0) and CuDNN (>=6.0). 

### Quick Start
1. Clone this repository.
2. Download the pre-trained model from this [link](https://drive.google.com/open?id=1u96zu0VyNpy-1VL90EbriN74hGaBBK08). Unzip it and put them into the "Models" folder.
3. Run "IntSeg_GUI.py", and a window will show up.
4. Open an image (one sample image is provided in "imgs"); the image will show on the up-left.
5. Use the mouse to input positive (left) and/or negative clicks.

The segmentation mask will show on the bottom-left, and the overlying image will show on the up-right. The bottom-right window can be ignored at this moment. The click inputs and segmentation results will be saved in the "res" folder under a random user id specified folder.

Note that the GUI is designed for demonstration only, and thus it is not optimized for images with arbitrary resolution.

## Citation
If you use our code for research, please cite our paper:

Zhuwen Li, Qifeng Chen, and Vladlen Koltun. Interactive Image Segmentation with Latent Diversity. In CVPR 2018.

## Todo List
1. Add the training code

## Question
If you have any question or request about the code and data, please email me at lzhuwen@gmail.com.

## License
MIT License
