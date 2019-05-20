import gym
import time
import tensorflow as tf
import numpy as np
import ddpg

env = gym.make('Pendulum-v0')

#target_actor.update(actor.model.trainable_variables)
#target_critic.update(critic.model.trainable_variables)

ou = ddpg.OrnsteinUhlenbeckActionNoise(mu=np.zeros(1))
buffer = ddpg.ReplayBuffer(100000)
global ep_ave_max_q_value
ep_ave_max_q_value = 0
global total_reward
total_reward = 0

def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

def train(actor, critic, action, reward, state, state2, done):
    global ep_ave_max_q_value

    actor.update_target_network()
    critic.update_target_network()

    buffer.add(np.reshape(state, (3,)),
        np.reshape(action, (2,)),
        reward,
        done,
        np.reshape(state2, (3,)))
    batch_size = 64

    if buffer.size() > batch_size:
        s_batch, a_batch, r_batch, t_batch, s2_batch = buffer.sample_batch(batch_size)

        # Calculate targets
        target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))

        yi = []
        #print(r_batch)
        for i in range(batch_size):
            if t_batch[i]:
                yi.append(r_batch[i])
            else:
                yi.append(r_batch[i] + 0.99 * target_q[i])

        ave_max_q, _ = critic.train(s_batch, a_batch, np.reshape(yi, (batch_size, 1)))

        ep_ave_max_q_value += np.amax(ave_max_q)
        
        a_outs = actor.predict(s_batch)
        grads = critic.action_gradients(s_batch, a_outs)
        actor.train(s_batch, grads[0])

        actor.update_target_network()
        critic.update_target_network()


with tf.Session() as sess:
    actor = ddpg.Actor(sess,  3, 2, np.array([-2, 2]), 0.0001, 0.001, 64)

    critic = ddpg.Critic(sess,  3, 2, 0.001, 0.001, 0.99, actor.get_num_trainable_vars())

    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())


    writer = tf.summary.FileWriter('./logs', sess.graph)


    for episode in range(10000):
        obs = env.reset()
        done = False
        j = 0
        ep_ave_max_q_value = 0
        total_reward = 0

        while not done:
            env.render()

            a = actor.predict(obs.reshape((1, 3))) + ou()

            obs2, reward, done, info = env.step(a[0])

            total_reward += reward

            train(actor, critic, a, reward, obs, obs2, done)
            obs = obs2
            j += 1
            #print(reward)
            #time.sleep(0.5)
        print('average_max_q: ', ep_ave_max_q_value / float(j), 'reward: ', total_reward, 'episode:', episode)
        summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: total_reward,
                    summary_vars[1]: ep_ave_max_q_value / float(j)
                })

        writer.add_summary(summary_str, episode)
        writer.flush()


env.close()
print('STOP')