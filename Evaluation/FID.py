from utils import inception_utils
from trainers import gan_trainer
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3
from scipy.stats import entropy


os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# calculate inception score with Keras
from math import floor
from numpy import ones
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import std
from numpy import exp
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input

# assumes images have the shape 299x299x3, pixels in [0,255]
def calculate_inception_score(images, n_split=10, eps=1E-16):
	# load inception v3 model
	model = InceptionV3()
	# convert from uint8 to float32

	processed = images.astype('float32')	# pre-process raw images for inception v3 model             
	processed = preprocess_input(processed)
	# predict class probabilities for images
	yhat = model.predict(processed)
	# enumerate splits of images/predictions
	scores = list()
	n_part = floor(images.shape[0] / n_split)
	for i in range(n_split):
		# retrieve p(y|x)
		ix_start, ix_end = i * n_part, i * n_part + n_part
		p_yx = yhat[ix_start:ix_end]
		# calculate p(y)
		p_y = expand_dims(p_yx.mean(axis=0), 0)
		# calculate KL divergence using log probabilities
		kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
		# sum over classes
		sum_kl_d = kl_d.sum(axis=1)
		# average over images
		avg_kl_d = mean(sum_kl_d)
		# undo the log
		is_score = exp(avg_kl_d)
		# store
		scores.append(is_score)
	# average across images
	is_avg, is_std = mean(scores), std(scores)
	return is_avg, is_std



def inception_score_funk(images, n_split=10, eps=1E-16):
    # Load InceptionV3 model
    model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

    # Prepare images for InceptionV3
    images = images.astype('float32')
    images = tf.image.resize(images, (299, 299))
    images = tf.keras.applications.inception_v3.preprocess_input(images)

    # Get predictions from InceptionV3
    preds = model.predict(images)

    # Split predictions into n_split parts
    split_preds = np.array_split(preds, n_split)

    # Compute the mean of softmax of each split of predictions
    scores = []
    for i, preds in enumerate(split_preds):
        kl = preds * (np.log(preds + eps) - np.log(np.expand_dims(np.mean(preds, axis=0), axis=0) + eps))
        kl = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl))

    # Compute the mean and standard deviation of the scores
    mean_score, std_score = np.mean(scores), np.std(scores)

    return mean_score, std_score





def fid():
    eval_steps=1
    strategy = tf.distribute.MirroredStrategy()
    real_images = tf.keras.preprocessing.image_dataset_from_directory(
      '/bhome/ovier/master/hit-gan/data/Size_256/One_image',
      labels='inferred',
      label_mode = "int",
      color_mode='rgb',
      image_size=(299,299))

    fake_images = tf.keras.preprocessing.image_dataset_from_directory(
      '/bhome/ovier/master/hit-gan/exp/exp15/Generated_images/Generated_2_run_last',
      labels='inferred',
      label_mode = "int",
      color_mode='rgb',
      image_size=(299, 299))
      
    inception_model = inception_utils.restore_inception_model()
    activations, _ = inception_utils.run_inception_model(
            real_images, inception_model, steps=eval_steps, strategy=strategy)

    gen_activations, gen_logits = inception_utils.run_inception_model(
                fake_images,
                inception_model,
                steps=eval_steps,
                strategy=strategy,
                )

    frechet_inception_distance = inception_utils.frechet_inception_distance(
                activations, gen_activations)
    #inception_score, x = inception_utils.inception_score(fake_images,32,num_images_per_split=1000)
    fake_images = np.concatenate([images for images, labels in fake_images], axis=0)
    inception_score, x = calculate_inception_score(fake_images)
    print(f'FID score is : {frechet_inception_distance} and the Inception score is: {inception_score} std: {x}')
    return frechet_inception_distance, inception_score, x

text_file = open("/bhome/ovier/master/hit-gan/exp/exp15/FID_and_IS/2_run_last.txt", "w")
fid_list = []
is_list = []
x_list = []
for i in range(100):   
  fid_score, inception_score, x = fid()
  text_file.write(f' FID SCORE IS: {fid_score} and the Inception score is: {inception_score} with a std on: {x} \n  ')
  fid_list.append(fid_score)
  is_list.append(inception_score)
  x_list.append(x)
text_file.write(f'Mean FID score is: {np.mean(fid_list)} and the mean is score is: {np.mean(is_list)} with a mean std on: {np.mean(x_list)}')
text_file.close()



