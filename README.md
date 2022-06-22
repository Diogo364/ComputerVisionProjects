# OpenCV Experiments

- [OpenCV Experiments](#opencv-experiments)
  - [About](#about)
  - [Quickstart](#quickstart)
    - [Requirements](#requirements)
    - [Runing the detector](#runing-the-detector)
    - [CLI Params](#cli-params)
  - [Implementing Your Image Transformation](#implementing-your-image-transformation)
    - [Extends the Interface](#extends-the-interface)
    - [Create an execution code](#create-an-execution-code)
    - [References](#references)

## About
This is a workspace environment created to facilitate the experimentation and learning process on the `OpenCV` library and Computer Vision field. To achieve that, this code structure was built using a few of the best pratices in Software Engineeiring, promoting code reuse and an easy ready-to-use Image Transformation Interface.

## Quickstart
Each code within the root of this project corresponds to a test using image or video. To demonstrate how this repo can be used the `caffe_detector.py` script will be used.

### Requirements
1. The Conda env used for development was exported to the `environment.yml` file. Run the command below to create your env:
    ```
    conda env create -f environment.yml
    ```
2. To run the `Caffe Face Detection` built-in in the OpenCV library it is necessary to download two files:
   - [ ] `.prototxt`: File containing the structure of the model;
   - [ ] `.caffemodel`: File containing the weights of each layer.
   > Both of these files can be obtained following the instructions in the `dnn` modeule from `opencv` Github repo, however, to facilitate that, they are available in [this Google Drive Link](https://drive.google.com/drive/folders/1DdP-3rQfoNYBIjjyZM02oGKwvU5XbPQC?usp=sharing).

### Runing the detector
- To test the Face Detector in a image just run the comand below:
    ```
    python caffe_detector.py -p path/to/.prototxt -m path/to/.caffemodel -i path/to/testimage
    ```
- To test the Face Detector in the webcam video just run the comand below:
    ```
    python caffe_detector.py -p path/to/.prototxt -m path/to/.caffemodel
    ```

### CLI Params
- `-p`, `--prototxt`: Path to the .prototxt file containing the model architecture.
- `-m`, `--model`: Path to the .caffemodel file containing the layers weights.
- `-c`, `--confidence`: Detection confidence threshold.
- `-i`, `--image_source`: Path to the test image. [For webcam do not pass this parameter].

## Implementing Your Image Transformation
These are the steps for you to use the power of this project's code modularization and create your own image transformation/preprocessing.

### Extends the Interface
Your Custom Image Transformation class needs to extends the `models/interface/imageTransformer.py` interface. This only requires that you Custom class to have a \_\_call__ method in which receives as input an image and generates another as output.
> Tip: As a good modularization manners, all Image Transformation classes are current in the `models` subdir.

### Create an execution code
Then, you must create your own execution code, in which you are going to import your Custom Image Transformation class and instantiate it will all required parameters before passing to the AppController.
> Tip: Follow the same example of the `caffe_detector.py`.

### References
- **OpenCV Github:** https://github.com/opencv/opencv
- **OpenCV Wiki:** https://github.com/opencv/opencv/wiki
- **OpenCV Models Instructions:** https://github.com/opencv/opencv/tree/4.x/samples/dnn
- **Code based on this PyImageSearch Tutorial:** https://pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/