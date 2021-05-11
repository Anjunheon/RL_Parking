# 원본 : https://github.com/HongJaeKwon/machine-learning/blob/master/07-RL/HJK_DDPG_2.ipynb
import tensorflow as tf
import gym
import numpy as np
import collections
import random
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class ContinuousCartPoleEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 30.0
        self.tau = 0.02  # seconds between state updates
        self.min_action = -1.0
        self.max_action = 1.0
        self.max_score = 300
        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(1,)
        )
        self.observation_space = spaces.Box(-high, high)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def stepPhysics(self, force):
        x, x_dot, theta, theta_dot = self.state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / \
                   (self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        return (x, x_dot, theta, theta_dot)

    def step(self, action):
        assert self.action_space.contains(action), \
            "%r (%s) invalid" % (action, type(action))
        # Cast action to float to strip np trappings
        force = self.force_mag * float(action)
        self.state = self.stepPhysics(force)
        x, x_dot, theta, theta_dot = self.state
        done = x < -self.x_threshold \
               or x > self.x_threshold \
               or theta < -self.theta_threshold_radians \
               or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("""
You are calling 'step()' even though this environment has already returned
done = True. You should always call 'reset()' once you receive 'done = True'
Any further steps are undefined behavior.
                """)
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None:
            return None

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))

    def close(self):
        if self.viewer:
            self.viewer.close()


env = ContinuousCartPoleEnv()
# problem = "Pendulum-v0"
# env = gym.make(problem)

# 2 - 액션 종류 슈 (아웃풋)
action_num=1
# 4 - 상태 종류 수 (인풋)
state_num=env.observation_space.shape[0]

i = tf.keras.layers.Input(shape=(state_num,))
out = tf.keras.layers.Dense(32, activation="relu")(i)
out = tf.keras.layers.BatchNormalization()(out)
out = tf.keras.layers.Dense(32, activation="relu")(out)
out = tf.keras.layers.BatchNormalization()(out)
out = tf.keras.layers.Dense(action_num, activation="tanh")(out)
u_model = tf.keras.Model(inputs=[i], outputs=[out])

i = tf.keras.layers.Input(shape=(state_num,))
out = tf.keras.layers.Dense(32, activation="relu")(i)
out = tf.keras.layers.BatchNormalization()(out)
out = tf.keras.layers.Dense(32, activation="relu")(out)
out = tf.keras.layers.BatchNormalization()(out)
out = tf.keras.layers.Dense(action_num, activation="tanh")(out)
t_u_model = tf.keras.Model(inputs=[i], outputs=[out])

i = tf.keras.layers.Input(shape=(state_num,))
out = tf.keras.layers.Dense(4, activation="relu")(i)
out = tf.keras.layers.BatchNormalization()(out)
out = tf.keras.layers.Dense(8, activation="relu")(out)
out = tf.keras.layers.BatchNormalization()(out)

a_i = tf.keras.layers.Input(shape=(action_num,))
a_out = tf.keras.layers.Dense(8, activation="relu")(a_i)
a_out = tf.keras.layers.BatchNormalization()(a_out)

out = tf.keras.layers.Concatenate()([out, a_out])

out = tf.keras.layers.Dense(32, activation="relu")(out)
out = tf.keras.layers.BatchNormalization()(out)
out = tf.keras.layers.Dense(32, activation="relu")(out)
out = tf.keras.layers.BatchNormalization()(out)
out = tf.keras.layers.Dense(1)(out)

q_model = tf.keras.Model([i, a_i], out)

i = tf.keras.layers.Input(shape=(state_num,))
out = tf.keras.layers.Dense(4, activation="relu")(i)
out = tf.keras.layers.BatchNormalization()(out)
out = tf.keras.layers.Dense(8, activation="relu")(out)
out = tf.keras.layers.BatchNormalization()(out)

a_i = tf.keras.layers.Input(shape=(action_num,))
a_out = tf.keras.layers.Dense(8, activation="relu")(a_i)
a_out = tf.keras.layers.BatchNormalization()(a_out)

out = tf.keras.layers.Concatenate()([out, a_out])

out = tf.keras.layers.Dense(32, activation="relu")(out)
out = tf.keras.layers.BatchNormalization()(out)
out = tf.keras.layers.Dense(32, activation="relu")(out)
out = tf.keras.layers.BatchNormalization()(out)
out = tf.keras.layers.Dense(1)(out)

t_q_model = tf.keras.Model([i, a_i], out)

t_u_model.set_weights(u_model.get_weights())
t_q_model.set_weights(q_model.get_weights())

opt=tf.keras.optimizers.Adam(0.001,clipnorm=0.1)

opt_2=tf.keras.optimizers.Adam(0.002,clipnorm=0.1)

tf.keras.utils.plot_model(u_model,show_shapes=True,show_layer_names=True)

tf.keras.utils.plot_model(q_model,show_shapes=True,show_layer_names=True)

scores = []
for episode in range(1000):
    done = False
    total_reward=0
    state = env.reset()
    state = np.reshape(state, [1, state_num])
    while not done:
        next_state, reward, done, _=env.step(env.action_space.sample())
        total_reward += reward
    scores.append(total_reward)

import matplotlib.pyplot as plt
plt.plot(scores)
print(np.mean(scores))

# 에피소드 수만큼 학습
episode_count = 1000

# 점수를 기록할 리스트
scores = []

# 디스카운트 팩터 정의
discount_rate = 0.99
batch_size = 256
tau = 0.1

# eps_mean=0.
# eps_std=0.1
# eps_theta=0.1
# eps_dt=0.01
# eps=0

state_list = collections.deque(maxlen=1000)
action_list = collections.deque(maxlen=1000)
reward_list = collections.deque(maxlen=1000)
next_state_list = collections.deque(maxlen=1000)
done_list = collections.deque(maxlen=1000)

for episode in range(episode_count):
    state = env.reset()
    # 차원을 맞추어 준다

    done = False
    total_reward = 0
    while not done:
        # noise=eps + eps_theta*(eps_mean-eps)*eps_dt + eps_std*np.sqrt(eps_dt)*np.random.normal(size=1)
        noise = np.random.normal(size=1)
        _state = np.reshape(state, [1, state_num])
        action = u_model.predict(_state)[0] + noise
        # eps=noise

        action = np.clip(action, -1.0, 1.0)

        next_state, reward, done, _ = env.step(action)
        state_list.append(state)
        action_list.append(action)
        reward_list.append(reward)
        next_state_list.append(next_state)
        done_list.append(1 - done)

        total_reward += reward
        state = next_state

        if len(state_list) >= batch_size:
            sample = random.sample(range(len(state_list)), batch_size)
            _state_list = tf.convert_to_tensor(np.array(state_list)[sample])
            _action_list = tf.convert_to_tensor(np.array(action_list)[sample])
            _reward_list = tf.convert_to_tensor(np.array(reward_list, dtype='float32')[sample])
            _done_list = tf.convert_to_tensor(np.array(done_list, dtype='float32')[sample])
            _next_state_list = tf.convert_to_tensor(np.array(next_state_list)[sample])

            with tf.GradientTape() as tape:
                q = q_model([_state_list, _action_list])
                n_a = t_u_model(_next_state_list)
                n_q = t_q_model([_next_state_list, n_a])
                tde = _reward_list + discount_rate * _done_list * tf.reshape(n_q, (batch_size,)) - tf.reshape(q, (
                batch_size,))
                q_loss = tf.math.square(tde)
                q_loss = tf.math.reduce_mean(q_loss)
            q_grad = tape.gradient(q_loss, q_model.trainable_variables)
            opt_2.apply_gradients(zip(q_grad, q_model.trainable_variables))

            with tf.GradientTape() as tape:
                a = u_model(_state_list)
                q = q_model([_state_list, a])
                l = -tf.math.reduce_mean(q)
            u_grad = tape.gradient(l, u_model.trainable_variables)
            opt.apply_gradients(zip(u_grad, u_model.trainable_variables))

        if (total_reward >= 200):
            done = True

        new_weights = []
        target_variables = t_q_model.weights
        for i, variable in enumerate(q_model.weights):
            new_weights.append(variable * tau + target_variables[i] * (1 - tau))

        t_q_model.set_weights(new_weights)

        new_weights = []
        target_variables = t_u_model.weights
        for i, variable in enumerate(u_model.weights):
            new_weights.append(variable * tau + target_variables[i] * (1 - tau))

        t_u_model.set_weights(new_weights)

    scores.append(total_reward)
    print(episode + 1, total_reward)

env.close()

import matplotlib.pyplot as plt
plt.plot(scores)
print(np.mean(scores))
