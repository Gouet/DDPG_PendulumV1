import tensorflow as tf
tf.enable_eager_execution()
from collections import deque
import random
import numpy as np


class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences 
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size: 
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0][0] for _ in batch], dtype='float32')
        a_batch = np.array([_[1][0] for _ in batch], dtype='float32')
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4][0] for _ in batch], dtype='float32')

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class Actor():
    def __init__(self, filename='./saved/actor.h5'):
        self.filename = filename

        w = tf.initializers.random_uniform(-0.003, 0.003)
        self.input = tf.keras.layers.Input(shape=(3,))

        x = tf.keras.layers.Dense(400)(self.input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(tf.nn.relu)(x)
        x = tf.keras.layers.Dense(300)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(tf.nn.relu)(x)

        out = tf.keras.layers.Dense(1, activation='tanh', kernel_initializer=w, bias_initializer=w)(x)
        out = tf.keras.layers.Lambda(lambda x: x * 2)(out)

        self.model = tf.keras.Model(inputs=[self.input], outputs=out)

        self.batch_size = 64

        self.model.summary()

        self.optimizer = tf.train.AdamOptimizer(0.0001)

    def save(self):
        self.model.save_weights(self.filename)

    def load(self):
        self.model.load_weights(self.filename)

    def train_step(self, state, action_gradients):
        newState = tf.constant(state)
        with tf.GradientTape() as tape:
            predictions = self.model(newState)
            
        gradient = tape.gradient(predictions, self.model.trainable_variables, -action_gradients)

        actor_gradients = list(map(lambda x: tf.math.divide(x, self.batch_size), gradient))

        self.optimizer.apply_gradients(zip(actor_gradients, self.model.trainable_variables))

class Critic():
    def __init__(self, filename='./saved/critic.h5'):
        self.filename = filename
        w = tf.initializers.random_uniform(-0.003, 0.003)
        self.input = tf.keras.layers.Input(shape=(3,))

        x = tf.keras.layers.Dense(400)(self.input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(tf.nn.relu)(x)
        x = tf.keras.layers.Dense(300)(x)

        self.input_action = tf.keras.layers.Input(shape=(1,))
        self.y = tf.keras.layers.Dense(300)(self.input_action)

        add = tf.keras.layers.Add()([x, self.y])
        merge = tf.keras.layers.Activation(tf.nn.relu)(add)

        out = tf.keras.layers.Dense(1, kernel_initializer=w, bias_initializer=w)(merge)

        self.model = tf.keras.Model(inputs=[self.input, self.input_action], outputs=out)

        self.batch_size = 64

        self.model.summary()

        self.optimizer = tf.train.AdamOptimizer(0.001)

    def train_step(self, state, action, predicted_q_value):
        newState = tf.constant(state)
        newAction = tf.constant(action)
        with tf.GradientTape() as tape:

            predictions = self.model([newState, newAction])

            loss = tf.losses.mean_squared_error(predictions, predicted_q_value)

        gradient = tape.gradient(loss, self.model.trainable_variables)

        self.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))
        return predictions

    def save(self):
        self.model.save_weights(self.filename)

    def load(self):
        self.model.load_weights(self.filename)

    def actor_gradient(self, state, actor):
        newState = tf.constant(state)
        with tf.GradientTape(persistent=True) as tape:
            actions = actor.model(newState)
            predictions = self.model([newState, actions])

        gradient = tape.gradient(predictions, actions)
        return gradient


class TargetActor(Actor):
    def __init__(self):
        super().__init__('./saved/target_actor.h5')
        self.tau = 0.001

    def hard_copy(self, actor_var):
        [self.model.trainable_variables[i].assign(actor_var[i])
                for i in range(len(self.model.trainable_variables))]

    def update(self, actor_var):
        [self.model.trainable_variables[i].assign(tf.multiply(actor_var[i], self.tau) \
            + tf.multiply(self.model.trainable_variables[i], 1. - self.tau))
                for i in range(len(self.model.trainable_variables))]

class TargetCritic(Critic):
    def __init__(self):
        super().__init__('./saved/target_critic.h5')
        self.tau = 0.001

    def hard_copy(self, critic_var):
        [self.model.trainable_variables[i].assign(critic_var[i])
                for i in range(len(self.model.trainable_variables))]

    def update(self, critic_var):
        [self.model.trainable_variables[i].assign(tf.multiply(critic_var[i], self.tau) \
            + tf.multiply(self.model.trainable_variables[i], 1. - self.tau))
                for i in range(len(self.model.trainable_variables))]

