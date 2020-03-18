'''
evaluation with test

one hidden layer

use reduce_mean for loss function: https://blog.csdn.net/xckkcxxck/article/details/80030111

lr: 0.005

align the dimension in loss 
    
use two hidden layers
'''
import tensorflow as tf 
import numpy as np
import gym
from inksci_module.memory import MEMORY
import matplotlib.pyplot as plt

class Agent():
    def __init__(self):
        self.env = gym.make("CartPole-v1")
        self.state_dim = 4
        self.action_dim = 2

        self.memory = MEMORY(10000, self.state_dim, self.action_dim)

        self.state_in, self.q_out = self.build_network()

        self.action_in, self.y_label, self.train_opt = self.create_train_method()

        self.sess = tf.Session()
        self.sess.run( tf.initialize_all_variables() )

        self.reward_list = []

        for _ in range(800):
            self.run_env()
            self.test_env()
        self.env.close()

        self.plot_reward()
        print(self.memory.pointer)
    def run_env(self):
        env = self.env
        observation = env.reset()
        for _ in range(1000):
          # env.render()
          action = self.noise_action(observation)

          observation_next, reward, done, info = env.step(action)

          self.memory.add(observation, self.one_hot(action, self.action_dim), reward, observation_next, done)
          if self.memory.pointer>1000:
            if self.memory.pointer<1002:
                print("train ...")
            self.train_network()
          observation = observation_next

          if done:
            break   

    def test_env(self):
        env = self.env
        observation = env.reset()
        reward_ep = 0
        for _ in range(1000):
          # env.render()
          action = self.get_action(observation)

          observation_next, reward, done, info = env.step(action)
          reward_ep += reward
          
          observation = observation_next

          if done:
            break   
        print("reward_ep: ", reward_ep)
        self.reward_list.append( reward_ep )    
    def build_network(self):

        # layers of network be 3 or more, will it works?

        x = tf.placeholder(tf.float32, shape=[None,self.state_dim])

        dense1 = tf.layers.dense(inputs=x, units=128, activation=tf.nn.relu)
        dense2 = tf.layers.dense(inputs=dense1, units=128, activation=tf.nn.relu)
        q = tf.layers.dense(inputs=dense2, units=self.action_dim, activation=None)    

        return x, q
    def noise_action(self, state):

        self.noise = 0.5

        if np.random.rand()<self.noise:
            action = np.random.randint(2)
        else:
            action = self.get_action(state)
        return action
    def get_action(self, state):
        fetches = {
            "q_out": self.q_out
        }
        feed_dict = {
            self.state_in: [state]
        }
        results = self.sess.run(fetches, feed_dict=feed_dict)
        action = np.argmax( results["q_out"][0] )    
        return action
    def create_train_method(self):
        # no matter how many the action are, just train q value of one action

        action_in = tf.placeholder(tf.float32, shape=[None,self.action_dim])

        y = tf.placeholder(tf.float32, shape=[None])

        q = tf.reduce_sum( tf.multiply(self.q_out, action_in), axis=1 )

        loss = tf.reduce_mean(tf.square(q-y))
        train_opt = tf.train.AdamOptimizer(0.005).minimize(loss)

        return action_in, y, train_opt
    def train_network(self):
        # create data for training
        pre_states, actions, rewards, post_states, dones = self.memory.sample(32)
        y = np.zeros( rewards.shape[0] )
        for i in range( rewards.shape[0] ):
            if dones[i]:
                y[i] = rewards[i]
            else:
                fetches = {
                    "q_out": self.q_out
                }
                feed_dict = {
                    self.state_in: [ post_states[i] ]
                }
                results = self.sess.run(fetches, feed_dict=feed_dict)

                y[i] = rewards[i] + 0.95*np.max( results["q_out"][0] )

        # train the network
        fetches = {
            "train": self.train_opt
        }
        feed_dict = {
            self.state_in: pre_states,
            self.action_in: actions,
            self.y_label: y
        }
        results = self.sess.run(fetches, feed_dict=feed_dict)

    def one_hot(self, action, action_dim):
        a = np.zeros(action_dim)
        a[action] = 1
        return a

    def plot_reward(self):
        # plt.plot( self.reward_list ,color="Blue", marker="^")
        # plt.savefig("dqn_ink7.png")

        fig = plt.figure()
        plt.scatter(range(len(self.reward_list)), self.reward_list, color='green', marker='^') # plot points
        plt.savefig("dqn_ink8.png")

agent = Agent()