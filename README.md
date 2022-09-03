# Object Detection in an Urban Environment

## Data

For this project, we will be using data from the [Waymo Open dataset](https://waymo.com/open/).

[OPTIONAL] - The files can be downloaded directly from the website as tar files or from the [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files/) as individual tf records. We have already provided the data required to finish this project in the workspace, so you don't need to download it separately.

## Structure

### Data

The data you will use for training, validation and testing is organized as follow:

```
/home/workspace/data/waymo
    - training_and_validation - contains 97 files to train and validate your models
    - train: contain the train data (empty to start)
    - val: contain the val data (empty to start)
    - test - contains 3 files to test your model and create inference videos
```

The `training_and_validation` folder contains file that have been downsampled: we have selected one every 10 frames from 10 fps videos. The `testing` folder contains frames from the 10 fps video without downsampling.

```
You will split this `training_and_validation` data into `train`, and `val` sets by completing and executing the `create_splits.py` file.


### Experiments
The experiments folder will be organized as follow:
```

experiments/
    - pretrained_model/
    - exporter_main_v2.py - to create an inference model
    - model_main_tf2.py - to launch training
    - reference/ - reference training with the unchanged config file
    - experiment0/ - create a new folder for each experiment you run
    - experiment1/ - create a new folder for each experiment you run
    - experiment2/ - create a new folder for each experiment you run
    - label_map.pbtxt
    ...

```
## Prerequisites

### Local Setup

For local setup if you have your own Nvidia GPU, you can use the provided Dockerfile and requirements in the [build directory](./build).

Follow [the README therein](./build/README.md) to create a docker container and install all prerequisites.

### Download and process the data

**Note:** ”If you are using the classroom workspace, we have already completed the steps in the section for you. You can find the downloaded and processed files within the `/home/workspace/data/preprocessed_data/` directory. Check this out then proceed to the **Exploratory Data Analysis** part.

The first goal of this project is to download the data from the Waymo's Google Cloud bucket to your local machine. For this project, we only need a subset of the data provided (for example, we do not need to use the Lidar data). Therefore, we are going to download and trim immediately each file. In `download_process.py`, you can view the `create_tf_example` function, which will perform this processing. This function takes the components of a Waymo Tf record and saves them in the Tf Object Detection api format. An example of such function is described [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#create-tensorflow-records). We are already providing the `label_map.pbtxt` file.

You can run the script using the following command:
```

python download_process.py --data_dir {processed_file_location} --size {number of files you want to download}

```
You are downloading 100 files (unless you changed the `size` parameter) so be patient! Once the script is done, you can look inside your `data_dir` folder to see if the files have been downloaded and processed correctly.

### Classroom Workspace

In the classroom workspace, every library and package should already be installed in your environment. You will NOT need to make use of `gcloud` to download the images.

## Instructions

### Exploratory Data Analysis

You should use the data already present in `/home/workspace/data/waymo` directory to explore the dataset! This is the most important task of any machine learning project. To do so, open the `Exploratory Data Analysis` notebook. In this notebook, your first task will be to implement a `display_instances` function to display images and annotations using `matplotlib`. This should be very similar to the function you created during the course. Once you are done, feel free to spend more time exploring the data and report your findings. Report anything relevant about the dataset in the writeup.

Keep in mind that you should refer to this analysis to create the different spits (training, testing and validation).


### Create the training - validation splits
In the class, we talked about cross-validation and the importance of creating meaningful training and validation splits. For this project, you will have to create your own training and validation sets using the files located in `/home/workspace/data/waymo`. The `split` function in the `create_splits.py` file does the following:
* create three subfolders: `/home/workspace/data/train/`, `/home/workspace/data/val/`, and `/home/workspace/data/test/`
* split the tf records files between these three folders by symbolically linking the files from `/home/workspace/data/waymo/` to `/home/workspace/data/train/`, `/home/workspace/data/val/`, and `/home/workspace/data/test/`

Use the following command to run the script once your function is implemented:
```

python create_splits.py --data-dir /home/workspace/data

```
### Edit the config file

Now you are ready for training. As we explain during the course, the Tf Object Detection API relies on **config files**. The config that we will use for this project is `pipeline.config`, which is the config for a SSD Resnet 50 640x640 model. You can learn more about the Single Shot Detector [here](https://arxiv.org/pdf/1512.02325.pdf).

First, let's download the [pretrained model](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz) and move it to `/home/workspace/experiments/pretrained_model/`.

We need to edit the config files to change the location of the training and validation files, as well as the location of the label_map file, pretrained weights. We also need to adjust the batch size. To do so, run the following:
```

python edit_config.py --train_dir /home/workspace/data/train/ --eval_dir /home/workspace/data/val/ --batch_size 2 --checkpoint /home/workspace/experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map /home/workspace/experiments/label_map.pbtxt

```
A new config file has been created, `pipeline_new.config`.

### Training

You will now launch your very first experiment with the Tensorflow object detection API. Move the `pipeline_new.config` to the `/home/workspace/experiments/reference` folder. Now launch the training process:
* a training process:
```

python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config

```
Once the training is finished, launch the evaluation process:
* an evaluation process:
```

python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config --checkpoint_dir=experiments/reference/

```
**Note**: Both processes will display some Tensorflow warnings, which can be ignored. You may have to kill the evaluation script manually using
`CTRL+C`.

To monitor the training, you can launch a tensorboard instance by running `python -m tensorboard.main --logdir experiments/reference/`. You will report your findings in the writeup.

### Improve the performances

Most likely, this initial experiment did not yield optimal results. However, you can make multiple changes to the config file to improve this model. One obvious change consists in improving the data augmentation strategy. The [`preprocessor.proto`](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto) file contains the different data augmentation method available in the Tf Object Detection API. To help you visualize these augmentations, we are providing a notebook: `Explore augmentations.ipynb`. Using this notebook, try different data augmentation combinations and select the one you think is optimal for our dataset. Justify your choices in the writeup.

Keep in mind that the following are also available:
* experiment with the optimizer: type of optimizer, learning rate, scheduler etc
* experiment with the architecture. The Tf Object Detection API [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) offers many architectures. Keep in mind that the `pipeline.config` file is unique for each architecture and you will have to edit it.

**Important:** If you are working on the workspace, your storage is limited. You may to delete the checkpoints files after each experiment. You should however keep the `tf.events` files located in the `train` and `eval` folder of your experiments. You can also keep the `saved_model` folder to create your videos.


### Creating an animation
#### Export the trained model
Modify the arguments of the following function to adjust it to your models:
```

python experiments/exporter_main_v2.py --input_type image_tensor --pipeline_config_path experiments/reference/pipeline_new.config --trained_checkpoint_dir experiments/reference/ --output_directory experiments/reference/exported/

```
This should create a new folder `experiments/reference/exported/saved_model`. You can read more about the Tensorflow SavedModel format [here](https://www.tensorflow.org/guide/saved_model).

Finally, you can create a video of your model's inferences for any tf record file. To do so, run the following command (modify it to your files):
```

python inference_video.py --labelmap_path label_map.pbtxt --model_path experiments/reference/exported/saved_model --tf_record_path /data/waymo/testing/segment-12200383401366682847_2552_140_2572_140_with_camera_labels.tfrecord --config_path experiments/reference/pipeline_new.config --output_path animation.gif

## Submission Write Up

### Project overview

This section should contain a brief description of the project and what we are trying to achieve. Why is object detection such an important component of self driving car systems?

The goal of this project is predicting the vehicle, pedestrian, cyclist, from recorded image of front camera of car. The object detection using  the optical camera is most prospective way to make the car drive itself because sensor price is cheap than [LiDAR](https://velodynelidar.com/what-is-lidar/) sensor.

### Set up

This section should contain a brief description of the steps to follow to run the code for this repository.

Specially, the Deep Learning technology is used for the object detection. To implement that method, I first build the the [Convolutional Neural Network](https://en.wikipedia.org/wiki/Convolutional_neural_network) model. Second, large collection dataset is needed to train and validate the built model. In this project, I use  the Perception dataset of [Waymo Open Dataset]([Open Dataset – Waymo](https://waymo.com/open/). The format of dataset is originally the *TFRecords* format which can used easily in the Deep Learning framework. The Tensorflow is selected for that. 

### Dataset

#### Dataset analysis

This section should contain a quantitative and qualitative description of the dataset. It should include images, charts and other visualizations.

1. Display dataset
   
   To explore dataset briefly, I first display the dataset as RGB image format like below. I can see that dadaset cosist of various weather, road, time condition. That means I need to select the data equally to train and evalution my CNN model.

![local image](eda_pic/eda_pic_11.png)

![local image](eda_pic/eda_pic_12.png)

2. RGB channel histogram
   
   To understand the dataset more deeply, I display the density of RGB channel graph format per 10 image. I can check that mean density of 10 images show similar distritition. Therefore, it has no problem to be used as dataset for the Neural Network.

![local image](eda_pic/eda_pic_16.png)

![local image](eda_pic/eda_pic_18.png)

#### 

#### Cross validation

This section should detail the cross validation strategy and justify your approach.

1. In this experiment, total 87 tfrecord are used for the train dataset. First, I display the class distritution of training dataset. 

![local image](eda_pic/eda_pic_26.png)

2. Next, I show the class distritution of 10 validation dataset. I can confirm that training and validation dataset has similair distritution. Thus, dataset has no issue to be used for CNN network.

![local image](eda_pic/eda_pic_27.png)

### Training

#### Reference experiment

I test the 3 kind of training strategy and compare the result using the loss, DetectionBoxes precision, DetectionBoxes recall.

```
> **Legend of Tensorboard**
- Default strategy: momentum optimizer, no data augumentation / learning rate decay

- Optimizer strategy: adam optimizer, learning rate decay, no data augumentation

- Final strategy: adam optimizer, learning rate decay, data augumentation
```

1. Loss
   
   Interestingly, the optimizer strategy is better than the final strategy at the loss metric. The default strategy is worst.
   
   ![local image](training_strategy_pic/loss_graph.png)

2. DetectionBoxes Precision
   
   Difference from the loss metric, the final strategy shows higher mAP score. The default strategy is worst again.
   
   ![local image](training_strategy_pic/precision_graph.png)

3. DetectionBoxes Recall
   
   Difference is little small than the precision metric, the final strategy is first rank in the DetectionBoxes Recall. The default strategy is last.
   
   ![local image](training_strategy_pic/recall_graph.png)

4. Visualizing detection result 
   
   The optimizer strategy works well for day time video. It does not make serious issue during driving.
   
   ![local image](optimizer_video/animation_1.gif)
   
   ![local image](optimizer_video/animation_2.gif)
   
   However, the result of night time looks not good. It miss the most of cars when they are far from the camera.  
   
   ![local image](optimizer_video/animation_3.gif)

#### Improve on the reference

There are several option I can use to improve the performance of my model. First thing is changing the optimizer which is used to update the weight of Neural Network.

1. **Optimizer setting**
   
   The default training strategy use the Momentum optimizer. From [post of online](https://medium.com/@ramrajchandradevan/the-evolution-of-gradient-descend-optimization-algorithm-4106a6702d39), I find the below animation which shows the speed of various optimizer finding the optimal point of loss function. I change the optimizer to Adam which use the positive side of the Momentum and Adagrad.
   
   ![local image](training_strategy_pic/optimizer_difference.gif)

2. **Data Augmentation**
   
   The next thing I consieder is generating more dataset by augumenting existence dataset. That can be implemented easily by adding the data_augmentation_options to config file of the object detection API of the Tensorflow.
   
   ```
      > **List of data augmentation method**
   RandomCropImage, NormalizeImage, RandomHorizontalFlip, 
   RandomRotation90, RandomBlackPatches, RandomAdjustBrightness, 
   RandomAdjustContrast, RandomAdjustHue, RandomAdjustSaturation, 
   RandomDistortColo
   ```
   
   ![local image](aug_pic/aug_pic_1.png)
   
   ![local image](aug_pic/aug_pic_4.png)
   
   ![local image](aug_pic/aug_pic_5.png)

3. **Decay the learning rate**
   
   The final method that is used in the Deep Learning area to improve the model is decaying the learning rate. The learning rate of each training strategy looks like a below graph.
   
   ![local image](training_strategy_pic/lr_graph.png)

#### Improved result after applying training technique

I test again new training strategy to night time video that failed before. Fortunately, it can detect the car of far location.

![local image](final_video/animation_3.gif)
