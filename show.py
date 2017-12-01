# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time

from environment.environment import Environment
from model.model import UnrealModel
from train.experience import Experience, ExperienceFrame
import pickle
import os.path
import cPickle as cp
import tensorflow as tf
from options import get_options

flags = get_options("show")

class Agent(object):
  def __init__(self,
               env_type,
               env_name,
               use_pixel_change,
               use_value_replay,
               use_reward_prediction,
               use_future_reward_prediction,
               use_autoencoder,
               reward_length,
               pixel_change_lambda,
               entropy_beta,
               device,
               skip_step):
    self.env_type = env_type
    self.env_name = env_name
    self.use_pixel_change = use_pixel_change
    self.use_value_replay = use_value_replay
    self.use_reward_prediction = use_reward_prediction
    self.use_future_reward_prediction = use_future_reward_prediction
    self.use_autoencoder = use_autoencoder
    self.skip_step = skip_step
    self.action_size = Environment.get_action_size(env_type, env_name)
    config = tf.ConfigProto(log_device_placement=False,
                            allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    self.sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    self.sess.run(init)
    self.network = UnrealModel(self.action_size,
                                     -1,
                                     use_pixel_change,
                                     use_value_replay,
                                     use_reward_prediction,
                                     use_future_reward_prediction,
                                     use_autoencoder,
                                     pixel_change_lambda,
                                     entropy_beta,
                                     device)
    self.network.prepare_loss()
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(flags.checkpoint_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(self.sess, checkpoint.model_checkpoint_path)
        print("checkpoint loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old checkpoint")
    #self.apply_gradients = grad_applier.minimize_local(self.network.total_loss,
    #                                                   global_network.get_vars(),
    #                                                   self.network.get_vars())
    
    #self.sync = self.network.sync_from(global_network)
    self.experience = Experience(10**4, reward_length)
    #self.local_t = 0
    self.episode_reward = 0
    # For log output
    #self.prev_local_t = 0
    #self.log_file = log_file
    #self.prediction_res_file = log_file + '/' + 'res.pkl'
    self.environment = Environment.create_environment(self.env_type,
                                                      self.env_name, self.skip_step, keep_raw_img=True)

  def stop(self):
    self.environment.stop()
  
  def choose_action(self, pi_values):
    return np.random.choice(range(len(pi_values)), p=pi_values)


  def _run_episode(self, epi_n, summary_writer=None, summary_op=None, score_input=None):
    # [Base A3C]
    states = []
    last_action_rewards = []
    actions = []
    rewards = []
    values = []
    #terminal_end = False

    start_lstm_state = self.network.base_lstm_state_out

    # t_max times loop
    for epi_i in range(epi_n):
      print(epi_i)
      while True:
        # Prepare last action reward
        last_action = self.environment.last_action
        last_reward = self.environment.last_reward
        last_action_reward = ExperienceFrame.concat_action_and_reward(last_action,
                                                                      self.action_size,
                                                                      last_reward)
        
        pi_, value_ = self.network.run_base_policy_and_value(self.sess,
                                                                   self.environment.last_state,
                                                                   last_action_reward)
        
        
        action = self.choose_action(pi_)

        states.append(self.environment.last_state)
        last_action_rewards.append(last_action_reward)
        actions.append(action)
        values.append(value_)

        prev_state = self.environment.last_state

        # Process game
        new_state, reward, terminal, pixel_change, raw_img = self.environment.process(action)
        frame = ExperienceFrame(prev_state, reward, action, terminal, pixel_change,
                                last_action, last_reward, raw_img=raw_img)

        # Store to experience
        self.experience.add_frame(frame)

        self.episode_reward += reward

        rewards.append( reward )

        if terminal:
          terminal_end = True
          print("score={}".format(self.episode_reward))

          #self._record_score(sess, summary_writer, summary_op, score_input,
          #                   self.episode_reward, global_t)
            
          self.episode_reward = 0
          self.environment.reset()
          self.network.reset_state()
          break

  def get_prediction(self, history, action):
      action_size = Environment.get_action_size(self.env_type, self.env_name)
      global_network = self.network
      feed_dict = {global_network.frp_input: history, #np.zeros((4, 84, 84, 3))
                   global_network.frp_action_input: action} #np.zeros((1, action_size)) fake frames and action input
      encoder_output = self.sess.run(global_network.encoder_output, feed_dict)
      print(encoder_output)

if __name__ == '__main__':
  device = "/cpu:0"
  agent = Agent(flags.env_type,
                flags.env_name,
                flags.use_pixel_change,
                flags.use_value_replay,
                flags.use_reward_prediction,
                flags.use_future_reward_prediction,
                flags.use_autoencoder,
                flags.reward_length,
                flags.pixel_change_lambda,
                flags.entropy_beta,
                device,
                flags.skip_step)
  config = tf.ConfigProto(log_device_placement=False,
                            allow_soft_placement=True)
  config.gpu_options.allow_growth = True
  agent._run_episode(1)
  with open('visualize_data/agent', 'w') as f:
    cp.dump(agent, f)
  rp_experience_frames, total_raw_reward, next_frame = agent.experience.sample_rp_sequence()
  history = []
  for i in range(4):
      history.append(rp_experience_frames[i].state)
  action_one_hot = np.zeros((1,agent.action_size))
  action_one_hot[0][rp_experience_frames[3].action] = 1
  encoder_output = agent.get_prediction(history, action_one_hot)
  with open('visualize_data/test_encoder_output', 'w') as f:
      cp.dump(encoder_output, f)



  
