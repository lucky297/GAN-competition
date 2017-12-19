# GAN-competition (Predicting Future Frames)
The project is worked by Yi-Chun Li, Wen-Chi Chin and Ting-Yi Hsieh.

## Introduction
### Main Idea and Novelty
Image processing has been studied for a long time. 
One of the well-known techniques is image-to-image translation, 
which achieves extraordinary results. 
We could convert images into the styles of famous painters, 
convert objects in the images into other objects or vice versa. 
However, it seems that future frames prediction is still a challenging task. 
Unlike image transfer, the complex appearance and motion dynamics of scenes limit the qualities of frames prediction. 
The existing methods [2, 3] mostly result in blurry predictions or weird-shaped objects in predictions.

Our project aims to improve the sharpness of future frames prediction. 
Previous methods try to solve this problem by focusing on generators. 
They generate two images, one for the main image, 
which can be foreground or the future frame according to different methods, 
and the other one for the supported image, which can be background or the future flow. 
By doing so, the predictions increase in clarity limitedly. 
Hence, we would like to try it in with cycleGAN. 

In cycleGAN, the reconstruction parts helps to generate better results. So we expect the cycle-consistency 
can help improving the sharpness of predictions.

### System Overview
Our project aims to predict the future frames of the black and white comic. 
Some previous  methods using video frames and pixel flows to predict the next motion of the real image [3]. 
The state-of-the-art techniques in Generative Adversarial Networks such as CycleGAN [1] 
is able to learn the mapping of one image domain to another image domain using unpaired image data. 
Therefore, we hope to use two cycleGAN architecture to achieve our goal. 
The first cycleGAN is to generate the real RGB image from the black and white comic. 
Then we take this real RGB image as input and use another cycleGAN to predict the future frame. 
Finally, we feed this future frame to the first cycleGAN to convert the style to black and white comic. 

## Process
We used the existing code([CycleGAN in TensorFlow](https://github.com/4Catalyzer/cyclegan/blob/master/README.md)) 
and changed its dataset.

Our project is divided into two parts. 
First, we train one cycleGAN to convert black and white comic into real RGB images. 
Second, we train the other cycleGAN to convert the current frame into the next frame.

### Collect datasets
* For the first part, we limited the motion prediction to play basketball. 
As the paper of cycleGAN mentions, 
the cycleGAN trained with horses and zebras does not perform well on the images which a man ride a horse. 
Therefore, we speculated that if the cycleGAN was trained with mixed actions, 
the result would have been bad. In order to improve the performance, 
we pick specific actions like playing basketball which means people are interacting with the ball in all the images. 
Since there is no existing dataset with playing basketball, 
we have to collected our own dataset. We collect 1500 images for both black and white comic 
and real RGB data and trained the first cycleGAN.

* For the second part, from the several basketball actions, we decide to predict shooting a basketball.
We collect 1200 images for both aiming actions and shooting actions.

### Train two cycleGANs (Results)
* For the first cycleGAN, we trained twice. 

	From the first results, we can see that cycleGAN catches the contour of the main targets from black and white comic images. 
	However, it’s difficult to show the exact appearance of people in the generated images. 
	
	We figured out some factors that might lead to such results. 
	
	* First, the complexity of background will influence the sharpness of predictions. 
		Since the actions playing basketball are various and complicated, 
		the contours of main targets for each images could be every different. 
		Hence, finding main targets could be a problem.	
		And if the background is full of noisy, cycleGAN would be hard to distinguish which is foreground and which is background. 
		Thus the foreground and the background are both blurry. 
	* Second, the exaggeration of comic makes it hard to show the details after converting into real images. 
		The difference between comic and real images is that comic sometimes exaggerate or 
		simplify the features and ignore the details. 
		Therefore, the cycleGAN has to learn to make up the details so that it can convert 
		something simple into something complicated. 
	* Third, silhouettes might be confused with background. 
		In the results, there are some generated images with white main target and colorful background 
		which inputs are silhouettes. 
		It seems that the cycleGAN mistook the background as foreground. 
		So it ignored the main target and focused on the background. 
	
	To solve the problems mentioned above, we simplify the images we have. 
	We make the backgrounds become simple to reduce the noisy and turn silhouettes into edges. 
	
	We then simplified some images and trained again. From the second results, we can see that cycleGAN not only catches the 
	contours of the main targets but also learns the details. We'll keep working on it.
 
* For the second cycleGAN, the results are pretty bad. 

	The cycleGAN didn't learn as same as what we expected. 
	Most of the actions would be changed in the next frames. However, the cycleGAN won't change the contours of the main targets.
	What it learns is how to fill up the main targets. Therefore, we're considering another way to train the cycleGAN.

	Since the cycleGAN seems to keep the original constructures of inputs, we decided to train the cycleGAN with optical flows. 
	Optical flows have the similar constructures with the inputs, which might be a feasible way to train cycleGAN. 
	We input aiming actions and try to output the optical flows. Then, we convert the optical flows into real images by 
	traditional wrapping methods. By doing so, we believe that predicting future frames can be success.
	
## Implementation
### Prepare datasets
* Create the data directory for two datasets in `input/`. 
(`basketball/` for the first cycleGAN and `actions/` for the second cycleGAN)

* Create the csv file as input to the data loader. 
	* Edit the cyclegan_datasets.py file.
	```python
	DATASET_TO_SIZES = {
    'basketball_train': 1500
	}

	PATH_TO_CSV = {
    'basketball_train': './input/basketball/basketball_train.csv'
	}

	DATASET_TO_IMAGETYPE = {
    'basketball_train': '.jpg'
	}
	``` 
	
	* Run create_cyclegan_dataset.py:
	```bash
	python create_cyclegan_dataset.py \
			--image_path_a=./input/basketball/trainA \
			--image_path_b=./input/basketball/trainB \
			--dataset_name="basketball_train" --do_shuffle=0
	```

### Training
* Create the configuration file. 

* Start training:
```bash
python main.py \
	--to_train=1 \
	--log_dir=./output/cyclegan/exp_01 \
	--config_filename=./configs/exp_01.json\
```
* Check the intermediate results. 
	* Tensorboard
	```bash
	tensorboard --port=6006 --logdir=./output/cyclegan/exp_01/#timestamp#
	```
	* Check the html visualizationat ./output/cyclegan/exp_01/#timestamp#/epoch_#id#.html.  

### Restoring from the previous checkpoint.
```bash
python main.py \
		--to_train=2 \
		--log_dir=./output/cyclegan/exp_01 \
		--config_filename=./configs/exp_01.json \
		--checkpoint_dir=./output/cyclegan/exp_01/#timestamp#
```
### Testing
* Create the testing dataset.
	* Edit the cyclegan_datasets.py file the same way as training.
	* Create the csv file as the input to the data loader. 
	```bash
	python create_cyclegan_dataset.py \
			--image_path_a=./input/basketball/testA \
			--image_path_b=./input/basketball/testB \
			--dataset_name="basketball_test" --do_shuffle=0
	```
* Run testing.
```bash
python main.py \
	--to_train=0 \
	--log_dir=./output/cyclegan/exp_01 \
	--config_filename=./configs/exp_01_test.json \
	--checkpoint_dir=./output/cyclegan/exp_01/#old_timestamp# 
```
The result is saved in ./output/cyclegan/exp_01/#new_timestamp#.

## Referemce
[1] [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
[2] [Generating Videos with Scene Dynamics](http://carlvondrick.com/tinyvideo/paper.pdf)
[3] [Dual Motion GAN for Future-Flow Embedded Video Prediction.](http://openaccess.thecvf.com/content_ICCV_2017/papers/Liang_Dual_Motion_GAN_ICCV_2017_paper.pdf)
