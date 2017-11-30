from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from environment.environment import Environment
from model.model import UnrealModel
from options import get_options

env_type = 'gym'
flags = get_options("visualize")
def get_prediction(history, action, env_name, check_dir):
    action_size = Environment.get_action_size(env_type, env_name)
    global_network = UnrealModel(action_size,
                                     -1,
                                     #flags.use_pixel_change,
                                     #flags.use_value_replay,
                                     #flags.use_reward_prediction,
                                     #flags.use_future_reward_prediction,
                                     #flags.use_autoencoder,
                                     False,
                                     False,
                                     False,
                                     True,
                                     True,
                                     .0,
                                     .0,
                                     "/cpu:0")
    config = tf.ConfigProto(log_device_placement=False,
                            allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)

    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(check_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("checkpoint loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old checkpoint")

    feed_dict = {global_network.frp_input:np.zeros((4,83,83,3)), global_network.frp_action_input:np.zeros((1,action_size))}
    encoder_output = sess.run(global_network.encoder_output, feed_dict)
    print(encoder_output)

if __name__ == '__main__':
    get_prediction(None, None, 'PongNoFrameskip-v0','/media/bighdd6/minghai1/capstone/results2/Pong_fsr_ae_l10/')