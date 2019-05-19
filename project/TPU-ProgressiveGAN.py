#!/usr/bin/env python3
# coding: utf-8

# In[ ]:


from datetime import datetime
from pytz import timezone

tz = timezone('EST')

print("Kernel is up")
print(datetime.now(tz))


# # WGAN-GP with DCGAN layers
# Code is mainly based upon the DCGAN implementation in the TensorFlow tutorials

# In[ ]:


import sys
sys.path.insert(0, '/home/asianzhang812_gmail_com/machine-learning-tone-generation/project/preprocessing')

import tensorflow as tf
import librosa
import os
import functools
import subprocess
import time
import numpy as np
import matplotlib.pyplot as plt
import math
import gc
from IPython import display
import time
import scipy.io.wavfile as wavfile
from tensorflow.layers import dense, flatten
from tensorflow.nn import relu, leaky_relu
from tensorflow import tanh
from tensorflow.image import ResizeMethod
import sys
import preprocessing.specgrams_helper as preprocessing

print("Finished imports")


# # Hyperparameters

# In[ ]:


# Number for large nsynth-train dataset
TOTAL_NUM = 102165
# Number for small nsynth-test dataset
# TOTAL_NUM = 1689
BUFFER_SIZE = 2048
batch_size = 64
PREFETCH_BUFFER_SIZE = 2 * batch_size
# EPOCHS = 150
gradient_penalty_weight = 10
real_score_penalty_weight = 0.001
ALPHA = 0.001
BETA1 = 0.0
BETA2 = 0.99
UPDATES_PER_GEN_UPDATE = 1
noise_dim = 100
num_steps = 100
num_tpu = 1
num_examples_to_generate = 16
epoch_counter = 0
kernel_size = 3
elements_per_stage_100k = 16.
spec_dim = (128, 1024, 2)
filters = [256, 256, 256, 256, 128, 64, 32]
epoch_proportion_counter = 0.0
model_dir = 'gs://jz-model-checkpoints/gan-tpu/'
#model_dir = 'gs://jz-model-checkpoints/gan-tpu-lr-001/'
tpu_name = "node2"


# # Defining models

# In[4]:


# Utility functions
def pixel_norm(images, epsilon=1.0e-8):
    return images * tf.rsqrt(tf.reduce_mean(tf.square(images), axis=3, keepdims=True) + epsilon)

def conv(x, filters, kernel_size, activation, name, padding='same'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        if activation == "pn_lrelu":
            activation = lambda x: pixel_norm(tf.nn.leaky_relu(x))
        elif activation == "lrelu":
            activation = tf.nn.leaky_relu
        elif activation is None:
            activation = lambda x: x

        if not isinstance(kernel_size, (list, tuple)):
            kernel_size = [kernel_size, kernel_size]
        kernel_size = list(kernel_size)
        kernel_shape = kernel_size + [x.shape.as_list()[3], filters]
        
        scale = np.sqrt(2. / (1. + np.prod(kernel_shape[:-1])))

        bias = tf.get_variable('bias', shape=(filters,), initializer=tf.zeros_initializer())
        
        return activation(scale * tf.layers.conv2d(x, filters, kernel_size, padding=padding, use_bias=False, kernel_initializer=tf.random_normal_initializer(stddev=1.0)) + bias) 

def dense(x, units, activation, name):
    with(tf.variable_scope(name, reuse=tf.AUTO_REUSE)):
        if activation is None: activation = lambda x : x
        bias = tf.get_variable('bias', shape=(units,), initializer=tf.zeros_initializer())
        kernel_scale=np.sqrt(2. / ((1. + 1.**2) * np.prod((x.shape.as_list()[-1], units)[:-1])))
        return activation(kernel_scale * tf.layers.dense(x, units, use_bias=False, kernel_initializer=tf.random_normal_initializer(stddev=1.)) + bias)

def to_rgb(x): 
    return conv(x, 2, 1, tf.nn.tanh, 'to_rgb')

def from_rgb(x, filters):
    return conv(x, filters, 1, 'lrelu', 'from_rgb')

def upscale(x, out_shape_or_scale=spec_dim[0:2]):
    with tf.variable_scope('nn_upscale'):
        shape = x.shape.as_list()
        if isinstance(out_shape_or_scale, (tuple, list)):
            out_shape = out_shape_or_scale
            scale = int(out_shape[0]/shape[1])
        else:
            scale = out_shape_or_scale
            out_shape = [shape[1]*scale, shape[2]*scale]
        filters = tf.tile(tf.expand_dims(tf.expand_dims(tf.Variable(lambda: tf.eye(shape[3]), trainable=False, name='filters'), 0), 0), [scale, scale, 1, 1])
        if(len(out_shape)==2):
            out_shape = [shape[0], out_shape[0], out_shape[1], shape[3]]
        return tf.nn.conv2d_transpose(x, filters, out_shape, [1, scale, scale, 1])

def downscale(x, scale=2):
    return tf.nn.avg_pool(x, [1, scale, scale, 1], [1, scale, scale, 1], 'VALID', name='avg_pool')

def upscale_conv(x, filters, id_num, kernel_size=kernel_size, scale=2):
    with tf.variable_scope('upscale_conv_{}'.format(id_num), reuse=tf.AUTO_REUSE):
        x = upscale(x, scale)
        x = conv(x, filters, kernel_size, 'pn_lrelu', name='conv_1')
        x = conv(x, filters, kernel_size, 'pn_lrelu', name='conv_2')
        return x
        
def conv_downscale(x, filters, id_num, kernel_size=kernel_size, scale=2):
    with tf.variable_scope("conv_downscale_{}".format(id_num), reuse=tf.AUTO_REUSE):
        x = conv(x, filters, kernel_size, 'lrelu', name='conv_1')
        x = conv(x, filters, kernel_size, 'lrelu', name='conv_2')
        x = tf.nn.avg_pool(x, [1, scale, scale, 1], [1, scale, scale, 1], 'VALID', name='avg_pool')
        return x
        
def generator_scale_schedule(filters):
    with tf.variable_scope('generator_scale_schedule', reuse=tf.AUTO_REUSE):
        global_step = tf.train.get_or_create_global_step()

        num_examples_100k = tf.cast(global_step, tf.float32) * tf.constant(batch_size / 100000., dtype=tf.float32)
        stage = elements_per_stage_100k
        out = [tf.clip_by_value(-tf.abs((num_examples_100k-i)*2/stage)+1.5, 0., 1.) for i in np.arange(stage/4, stage*(4*len(filters)+1)/4, stage)]
        return out

def discriminator_scale_schedule(filters):
    with tf.variable_scope('discriminator_scale_schedule', reuse=tf.AUTO_REUSE):
        global_step = tf.train.get_or_create_global_step()
        num_examples_100k = tf.cast(global_step, tf.float32) * tf.constant(batch_size / 100000., dtype=tf.float32)
        stage = elements_per_stage_100k
        out = [tf.clip_by_value((num_examples_100k-i)/stage*2, 0., 1.) for i in np.arange(stage/2, stage*(2*len(filters)+1)/2, stage)]
        return out


# In[5]:


def generator(x):
    # Input x is noise vector
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
        intermed = []
        alpha = generator_scale_schedule(filters)
        initial_shape = (int(np.round(spec_dim[0]/(2**(len(filters)-1)))), int(np.round(spec_dim[1]/(2**(len(filters)-1)))))
        with tf.variable_scope('project_block_0', reuse=tf.AUTO_REUSE):
            x = tf.layers.flatten(x)
            x = tf.expand_dims(tf.expand_dims(x, 1), 1)
            x = pixel_norm(x)
            x = tf.pad(x, [[0] * 2, [initial_shape[0] - 1] * 2, [initial_shape[1] - 1] * 2, [0] * 2])
            x = conv(x, filters[0], (initial_shape[0], initial_shape[1]), 'pn_lrelu', 'expand', padding='VALID')
            x = conv(x, filters[0], kernel_size, 'pn_lrelu', 'conv')
            intermed.append(to_rgb(x)*alpha[0])
        for i in range(1, len(filters)):
            with tf.variable_scope('block_{}'.format(i), reuse=tf.AUTO_REUSE):
                x = upscale_conv(x, filters[i], i)
                intermed.append(to_rgb(x)*alpha[i])
        with tf.variable_scope('upscale_sum', reuse=tf.AUTO_REUSE):
            ans_intermed = []
            for i in range(0, len(intermed)):
                with tf.variable_scope('block_{}'.format(i)):
                    ans_intermed += [upscale(intermed[i])]
            ans = tf.add_n(ans_intermed)
        return ans, intermed
        
        
    
def discriminator(x, intermed=None):
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        alpha = discriminator_scale_schedule(filters)
        gen_alpha = generator_scale_schedule(filters)
        if intermed is None:
            made_intermed = [x]
            while(len(made_intermed) != len(gen_alpha)):
                made_intermed = [downscale(made_intermed[0])] + made_intermed
        else:
            assert len(intermed) == len(gen_alpha)
            intermed = [gen_alpha[i]*intermed[i] for i in range(0, len(intermed))]
        original_x = x
        with tf.variable_scope("block_{}".format(len(filters)-1), reuse=tf.AUTO_REUSE):
            x = from_rgb(x, filters[-1])
        for i in range(len(filters)-1, 0, -1):
            with tf.variable_scope("block_{}".format(i), reuse=tf.AUTO_REUSE):
                x = conv_downscale(x, filters[i], i)
                if intermed:
                    to_be_combined = intermed[i-1]*gen + downscale(intermed[i])
                else:
                    to_be_combined = made_intermed[i-1]
                x = from_rgb(to_be_combined, filters[i])*(1-alpha[i-1]) + x*alpha[i-1]
        with tf.variable_scope("block_0", reuse=tf.AUTO_REUSE):
            with tf.variable_scope("batch_discrimination", reuse=tf.AUTO_REUSE):
                mean, var = tf.nn.moments(x, axes=[0])
                del mean
                x = tf.concat(
                    [x,
                    tf.ones([tf.shape(x)[i] for i in range(4 - 1)] + [1]) * tf.reduce_mean(tf.sqrt(var + 1e-6))],
                    axis=4 - 1)
            x = conv(x, filters[0], kernel_size, 'lrelu', 'conv_1')
            x = conv(x, filters[0], kernel_size, 'lrelu', 'conv_2')
            x = dense(x, 1, None, 'dense')
            assert x is not None
            return x


# # Making a TPUEstimator

# In[6]:


data_helper = preprocessing.SpecgramsHelper(spec_dim[0:2])
def record_parser(raw_data):
    read_features = {
        'note': tf.FixedLenFeature([], dtype=tf.int64),
        'note_str': tf.FixedLenFeature([], dtype=tf.string),
        'instrument': tf.FixedLenFeature([], dtype=tf.int64),
        'instrument_str': tf.FixedLenFeature([], dtype=tf.string),
        'pitch': tf.FixedLenFeature([], dtype=tf.int64),
        'velocity': tf.FixedLenFeature([], dtype=tf.int64),
        'sample_rate': tf.FixedLenFeature([], dtype=tf.int64),
        'spectrogram': tf.FixedLenFeature([262144], dtype=float),
        'instrument_family': tf.FixedLenFeature([], dtype=tf.int64),
        'instrument_family_str': tf.FixedLenFeature([], dtype=tf.string),
        'instrument_source': tf.FixedLenFeature([], dtype=tf.int64),
        'instrument_source_str': tf.FixedLenFeature([], dtype=tf.string)
    }
    
    data = tf.parse_single_example(serialized=raw_data, features=read_features)
    x = data['spectrogram']
    assert x is not None
    return tf.reshape(x, spec_dim)

def input_fn(params):
    with tf.variable_scope('input-pipeline'):
        batch_size = params['batch_size']
        # Reading features of TFRecord file
        files = tf.data.Dataset.list_files('gs://jz-datasets/spec-pruned-files/*.tfrecord')
        specs = files.apply(tf.data.experimental.parallel_interleave(tf.data.TFRecordDataset, cycle_length=2))
        specs = specs.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=BUFFER_SIZE))
        # specs = specs.map(map_func=(lambda raw_data: tf.reshape(tf.parse_single_example(serialized=raw_data, features=read_features)['spectrogram'], spec_dim)), num_parallel_calls=-1)
        specs = specs.apply(tf.data.experimental.map_and_batch(
            map_func=lambda x: (record_parser(x), tf.zeros(batch_size)), 
            num_parallel_calls=-1, 
            batch_size=batch_size, 
            drop_remainder=True))
        specs = specs.prefetch(buffer_size=2*batch_size)
        assert specs is not None
        return specs
    
def host_call_fn(global_step, generated_images, gen_cost, discrim_orig_cost, gradient_penalty, real_score_penalty, combined_discrim_loss):
    gs = global_step[0]
    audio = data_helper.melspecgrams_to_waves(generated_images)
    with tf.contrib.summary.create_file_writer(model_dir, max_queue=num_steps).as_default():
        with tf.contrib.summary.always_record_summaries():
            tf.contrib.summary.scalar('generator_cost', gen_cost[0], step=gs)
            tf.contrib.summary.scalar('discriminator_gan_loss', discrim_orig_cost[0], step=gs)
            tf.contrib.summary.scalar('gradient_penalty', gradient_penalty[0], step=gs)
            tf.contrib.summary.scalar('real_score_penalty', real_score_penalty[0], step=gs)
            tf.contrib.summary.scalar('combined_disciminator_loss', combined_discrim_loss[0], step=gs)
            tf.contrib.summary.image('log_magnitudes', generated_images[:, :, :, 0:1], step=gs)
            tf.contrib.summary.image('instantaneous_frequency', generated_images[:, :, :, 1:2], step=gs)
            tf.contrib.summary.audio('generated_sound', audio, 16000, 1, step=gs)
            
            return tf.contrib.summary.all_summary_ops()

def model_fn(features, labels, mode, params):
    assert features is not None
    assert labels is not None
    batch_size = params['batch_size']
    global_step = tf.train.get_or_create_global_step()
    run_discriminator = tf.ceil(tf.div(tf.cast(tf.mod(global_step, (UPDATES_PER_GEN_UPDATE+1)), tf.float32), float(UPDATES_PER_GEN_UPDATE+1)))
    with tf.variable_scope('runs'):
        real_images = features
        noise = tf.random_normal([batch_size, noise_dim])
        fake_images, fake_intermed = generator(noise)
        discriminator_real = discriminator(real_images)
        discriminator_fake = discriminator(fake_images)
        with tf.variable_scope('gradient-penalty'):
            alpha = tf.random_uniform(shape=[batch_size, spec_dim[0], spec_dim[1], spec_dim[2]], minval=0., maxval=1.)
            differences = fake_images-real_images
            interpolates = real_images+(alpha*differences)
        discriminator_interpolate = discriminator(interpolates)

    def restore_batch_size(x):
        return tf.tile(tf.reshape(x, [1, 1]), [batch_size, 1])
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        test_images = {
            'fake_images': fake_images,
            'real_images': real_images,
            'global_step': restore_batch_size(global_step),
        }
        return tf.contrib.tpu.TPUEstimatorSpec(mode, predictions=test_images)
    
    with tf.variable_scope('costs'):
        with tf.variable_scope('generator_cost'):
            pre_gen_cost = -tf.reduce_mean(discriminator_fake)
            gen_cost = pre_gen_cost*(1-run_discriminator)
        
        with tf.variable_scope('discriminator_cost'):
            original_cost = tf.reduce_mean(discriminator_fake)-tf.reduce_mean(discriminator_real)

            with tf.variable_scope('gradient-penalty'):
                gradients = tf.gradients(discriminator_interpolate, [interpolates])[0]
                slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
                gradient_penalty = tf.reduce_mean((slopes-1.)**2)
                

            with tf.variable_scope('real-score-penalty'):
                real_score_penalty = tf.reduce_mean(tf.square(discriminator_real))

            pre_discriminator_cost = original_cost + gradient_penalty * gradient_penalty_weight + real_score_penalty * real_score_penalty_weight
            
            discriminator_cost = pre_discriminator_cost*run_discriminator

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.contrib.tpu.TPUEstimatorSpec(mode, loss=0) # , eval_metric_ops=costs)
    
    def restore_batch_size(x):
        return tf.tile(tf.reshape(x, [1, 1]), [batch_size, 1])
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        with tf.variable_scope('optimizers'):
            gen_opt = tf.contrib.tpu.CrossShardOptimizer(tf.train.AdamOptimizer(ALPHA, BETA1, BETA2))
            assert gen_opt is not None
            discriminator_opt = tf.contrib.tpu.CrossShardOptimizer(tf.train.AdamOptimizer(ALPHA, BETA1, BETA2))
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                discriminator_opt = discriminator_opt.minimize(discriminator_cost, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='runs/discriminator'))
                assert discriminator_opt is not None
            with tf.control_dependencies([discriminator_opt]):
                gen_opt = gen_opt.minimize(gen_cost, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='runs/generator'))
                assert discriminator_opt is not None
            with tf.control_dependencies([gen_opt]):
                opt = tf.assign_add(global_step, 1)
            assert tf.tile(tf.reshape(global_step, [1, 1]), [batch_size, 1]) is not None
            assert fake_images is not None
            tensors_to_pass = [
                restore_batch_size(global_step),
                fake_images,
                restore_batch_size(pre_gen_cost),
                restore_batch_size(original_cost),
                restore_batch_size(gradient_penalty),
                restore_batch_size(real_score_penalty),
                restore_batch_size(pre_discriminator_cost)
            ]
            assert opt is not None
            return tf.contrib.tpu.TPUEstimatorSpec(mode, train_op=opt, host_call=(host_call_fn, tensors_to_pass), loss=discriminator_cost+gen_cost)
    return
    #return generator, gen_opt, discriminator_opt, real_images, test_images, ranEpoch, getEpoch, increment, merged, global_step


# In[7]:


def runOneEpoch(model):
    start = time.time()
    
    model.train(input_fn, steps=num_steps)
    
    predictions = next(iter(model.predict(input_fn, yield_single_examples=False)))
    display.clear_output(wait=True)
    print(datetime.now(tz))
    global_step = generate_images(predictions, source='fake')
    generate_images(predictions, source='real')
    print("Finished global step {} in {} sec".format(global_step, time.time()-start))
    gc.collect()


# In[8]:


def generate_images(images, source='fake', save=True):
    # make sure the training parameter is set to False because we
    # don't want to train the batchnorm layer when doing inference.
    
    if(source=='fake'):
        disp_images = images['fake_images']
    elif(source=='real'):
        disp_images = images['real_images']
    else:
        raise ValueError
    plt.title(source.capitalize()+" log-magnitudes")
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(np.transpose(disp_images[i, :, :, 0]) * 127.5, cmap="magma", origin="lower", aspect="auto")
        plt.axis('off')
    if(save):
        plt.savefig('images/image_at_{}_{}.png'.format(images['global_step'][0, 0], source))
    plt.show()
    
    plt.title(source.capitalize()+" instantaneous frequencies")
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(np.transpose(disp_images[i, :, :, 1]) * 127.5, cmap="magma", origin="lower", aspect="auto")
        plt.axis('off')
    plt.show()
    audio = data_helper.melspecgrams_to_waves(disp_images)[:, :, 0].eval(session=tf.Session()) * 100000
    audio = audio.astype(np.float32)
    for i in range(0, 4):
        display.display(display.Audio(audio[i, :], rate=16000))
    
    
    return images['global_step'][0, 0]


# # Running the model

# In[ ]:


cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
    tpu=[tpu_name], 
    zone="us-central1-f", 
    project="jz-cloud-test"
)


tpu_run_config = tf.contrib.tpu.RunConfig(
    cluster=cluster_resolver, 
    model_dir=model_dir,
    session_config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True),
    tpu_config=tf.contrib.tpu.TPUConfig(num_steps, 8)
)  

model = tf.contrib.tpu.TPUEstimator(
    model_fn=model_fn, 
    config=tpu_run_config, 
    use_tpu=True, 
    train_batch_size=batch_size, 
    predict_batch_size=16,
)


# In[ ]:


# Waits for another program to remove the .lock file before continuing execution
# !rm .lock
# Comment out next line to disable lock
# !touch .lock
i = 0
while True:
    if(os.path.isfile('.lock')):
        break
    runOneEpoch(model)


# In[ ]:


display.clear_output(wait=True)
print(datetime.now(tz))
global_step = generate_images(predictions, source='fake')
generate_images(predictions, source='real')


# In[ ]:


"""
print("Actually running")
predictions = next(iter(model.predict(input_fn, yield_single_examples=False)))
print("1")
global_step, critic_step, gen_step = generate_images(predictions, source='fake', save=False)
print("2")
generate_images(predictions, source='real', save=False)
print("3")
print("Global step: {}".format(global_step))
testAudio(predictions['real_images'], "real")
testAudio(predictions['fake_images'], "fake-{}".format(global_step))
display.display(display.Audio("audio/real-1.wav"), display.Audio("audio/fake-{}-1.wav".format(global_step)))
"""


# In[ ]:


while(kernel is dead):
    plant_a_new_seed()


# In[5]:


# Trying to implement tf.image.resize_nearest_neighbors because TPU doesn't support that op
# Luckily I figured it out with a transposed convolution

"""
import tensorflow as tf
tf.enable_eager_execution()

x = [[1., 2., 3., 4., 5.],
     [6., 7., 8, 9, 10],
     [11., 12., 13., 14., 15.],
     [16., 17., 18., 19., 20.],
     [21., 22., 23., 24., 25]]
mut_x = list(x)
mut_x[0] = [26, 2, 3, 4, 5]
print(mut_x)
x = tf.constant([x, mut_x, x])
x = tf.transpose(x, [1, 2, 0])
print(x.shape)
x = tf.expand_dims(x, 0)
x = tf.Variable(x, dtype=tf.float32)
print(x.numpy()[0, :, :, 0])
print(x.numpy()[0, :, :, 1])

# filters = tf.ones([2, 2, 3, 3])
filters = tf.eye(3)
filters = tf.expand_dims(filters, 0)
filters = tf.expand_dims(filters, 1)
filters = tf.tile(filters, [scale, scale, 1, 1])

print(filters)

print(x[0, :, :, 0])
print(x[0, :, :, 1])
print(x[0, :, :, 2])
x = upscale(x, 4)
print(x[0, :, :, 0])
print(x[0, :, :, 1])
print(x[0, :, :, 2])
"""


# In[ ]:




