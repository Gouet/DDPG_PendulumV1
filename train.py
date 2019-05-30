import gym
import time
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
import ddpg
import os

try:  
    os.mkdir('./saved')
except OSError:  
    print ("Creation of the directory failed")
else:  
    print ("Successfully created the directory")

print(tf.__version__)
env = gym.make('Pendulum-v0')

critic = ddpg.Critic()
actor = ddpg.Actor()
target_critic = ddpg.TargetCritic()
target_actor = ddpg.TargetActor()

try:
    critic.load()
    actor.load()
except Exception as e:
    print(e.__repr__)

target_actor.hard_copy(actor.model.trainable_variables)
target_critic.hard_copy(critic.model.trainable_variables)

ou = ddpg.OrnsteinUhlenbeckActionNoise(mu=np.zeros(1,))
buffer = ddpg.ReplayBuffer(100000)
global ep_ave_max_q_value
ep_ave_max_q_value = 0
global total_reward
total_reward = 0

def create_tensorboard():
    global_step = tf.train.get_or_create_global_step()

    logdir = "./logs/"
    writer = tf.contrib.summary.create_file_writer(logdir)
    writer.set_as_default()
    return global_step, writer

global global_step
global_step, writer = create_tensorboard()

def train(action, reward, state, state2, done):
    global ep_ave_max_q_value
    
    buffer.add(state, action, reward, done, state2)
    batch_size = 64

    if buffer.size() > batch_size:
        s_batch, a_batch, r_batch, t_batch, s2_batch = buffer.sample_batch(batch_size)

        target_action2 = target_actor.model.predict(s2_batch)
        predicted_q_value = target_critic.model.predict([s2_batch, target_action2])

        yi = []
        for i in range(batch_size):
            if t_batch[i]:
                yi.append(r_batch[i])
            else:
                yi.append(r_batch[i] + 0.99 * predicted_q_value[i])

        predictions = critic.train_step(s_batch, a_batch, yi)

        ep_ave_max_q_value += np.amax(predictions)

        grad = critic.actor_gradient(s_batch, actor)
        actor.train_step(s_batch, grad)

        target_actor.update(actor.model.trainable_variables)
        target_critic.update(critic.model.trainable_variables)

for episode in range(10000):
    global_step.assign_add(1)

    obs = env.reset()
    done = False
    j = 0
    ep_ave_max_q_value = 0
    total_reward = 0
    while not done:
        env.render()
        obs = obs.reshape((1, 3))

        noise = ou()
        action = actor.model.predict(obs)

        action = action + noise

        obs2, reward, done, info = env.step(action)
        total_reward += reward

        train(action, reward, obs, obs2.reshape((1, 3)), done)
        obs = obs2
        j += 1

    with writer.as_default(), tf.contrib.summary.always_record_summaries():
        tf.contrib.summary.scalar("average_max_q", ep_ave_max_q_value / float(j))
        tf.contrib.summary.scalar("reward", total_reward)
    
    critic.save()
    actor.save()
        
    print('average_max_q: ', ep_ave_max_q_value / float(j), 'reward: ', total_reward, 'episode:', episode)

env.close()