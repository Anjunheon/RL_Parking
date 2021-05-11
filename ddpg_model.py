import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

import setup_path
import airsim
import time
import pyautogui
import pytesseract
from PIL import Image

num_states = 11
num_actions = 4

upper_bound = 1
lower_bound = 0

api_control = True

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(4, activation="tanh", kernel_initializer=last_init)(out)

    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic():
    # State as input
    state_input = layers.Input(shape=(num_states))
    state_out = layers.Dense(128, activation="relu")(state_input)
    state_out = layers.Dense(128, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(64, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model


def policy(state, noise_object):
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise

    np.clip(sampled_actions[3], 0, 1)
    sampled_actions[3] = 0 if sampled_actions[3] < 0.5 else 1

    legal_action = [np.clip(sampled_actions[0], 0, 1),  # brake
                    np.clip(sampled_actions[1], -1, 1),  # steering
                    np.clip(sampled_actions[2], -1, 1),  # throttle
                    sampled_actions[3]]  # direction

    return [np.squeeze(legal_action)]


def sim_start():  # 시뮬레이터 실행
    # print(pyautogui.position())  # (1125, 455)
    pyautogui.click(1125, 455)
    # time.sleep(1)

    pyautogui.keyDown('altleft')
    pyautogui.keyDown('p')
    pyautogui.keyUp('altleft')
    pyautogui.keyUp('p')
    time.sleep(1)

    pyautogui.click(1125, 455)

    # connect to the AirSim simulator
    client = airsim.CarClient()
    client.confirmConnection()
    client.enableApiControl(api_control)
    print("API Control enabled: %s\n" % client.isApiControlEnabled())
    car_controls = airsim.CarControls()

    time.sleep(1)

    return client, car_controls


def sim_stop():  # 시뮬레이터 중지
    # print(pyautogui.position())  # (1125, 455)
    pyautogui.click(1125, 455)
    time.sleep(1)

    # 시뮬레이터 종료
    pyautogui.keyDown('esc')
    pyautogui.keyUp('esc')
    time.sleep(1)


def capture_goal():  # 목표 지점의 언리얼 좌표 -> 에어심 좌표 변환
    # 언리얼에서 출력되는 목표 지점 좌표
    unreal_goals = [[600, 2600], [600, 2230], [600, 1800], [600, 1430], [600, 990], [600, 620],  # 우측
                    [-1200, 2600], [-1200, 2230], [-1200, 1800], [-1200, 1430], [-1200, 990]]  # 좌측

    # 에어심 API를 통해 출력되는 목표 지점 좌표
    airsim_goals = [[6, -14], [6, -17], [6, -22], [6, -25], [6, -30], [6, -33],  # 우측
                    [-7, -14], [-7, -17], [-7, -22], [-7, -25], [-7, -30]]  # 좌측

    # 좌표 출력 부분 스크린샷 캡쳐
    img = pyautogui.screenshot('goal.png', region=(36, 90, 210, 15))  # 전체화면(F11) 기준
    # 좌표 스크린샷 문자열로 변환
    goal_pos = pytesseract.image_to_string(Image.open('goal.png'))
    # print(goal_pos[:-2])
    
    # x, y 좌표 구분 -> 좌표 값 float 변환
    goal_pos = str.split(goal_pos[:-2], ' ')
 
    x = str.split(goal_pos[0], '.')[0]
    y = str.split(goal_pos[1], '.')[0]

    x = int(float(x[2:]))
    if y[0] == '¥':  # 가끔 문자를 잘못 인식하는 경우 발생
        y = int(float(y[3:]))
    else:
        y = int(float(y[2:]))

    goal_xy = []
    for i in range(len(airsim_goals)):
        if x == unreal_goals[i][0] and y == unreal_goals[i][1]:
            # print('Goal x :', airsim_goals[i][0])
            # print('Goal y :', airsim_goals[i][1])
            goal_xy = airsim_goals[i]
            print('Goal :', airsim_goals[i])
            break

    return goal_xy


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'


std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()

# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# Learning rate for actor-critic models
critic_lr = 0.002
actor_lr = 0.001

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

total_episodes = 10000
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005

buffer = Buffer(50000, 64)


# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

# 처음 실행 시 충돌 물체 인식 후 리셋 관련 문제 때문에 중지 후 다시 시작
_ = sim_start()
sim_stop()
client, car_controls = sim_start()

time.sleep(2)

# Takes about 4 min to train
for ep in range(total_episodes):

    # prev_state = env.reset()
    prev_state = [client.getCarState().kinematics_estimated.position.x_val,  # 차량 위치 x 좌표
                  client.getCarState().kinematics_estimated.position.y_val,  # 차량 위치 y 좌표
                  client.getCarState().speed,                                # 차량 속도
                  client.getCarControls().brake,                             # 브레이크
                  client.getCarControls().steering,                          # 핸들 방향
                  client.getCarControls().throttle,                          # 차량 이동
                  client.getCarControls().manual_gear,                       # 후진 기어
                  client.getDistanceSensorData("Distance1").distance,        # 전방 거리 센서
                  client.getDistanceSensorData("Distance2").distance,        # 우측 거리 센서
                  client.getDistanceSensorData("Distance3").distance,        # 후방 거리 센서
                  client.getDistanceSensorData("Distance4").distance]        # 좌측 거리 센서

    episodic_reward = 0

    is_captured = 0
    count = 0
    start_time = 0
    end_time = 0

    reward = 0
    done = False

    while True:
        if is_captured == 0:
            goal = capture_goal()
            is_captured = 1

        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        action = policy(tf_prev_state, ou_noise)
        action = tf.squeeze(action)

        print('episode :', ep, '|',
              'brake :', round(float(action[0]), 3), '|', 'steering :', round(float(action[1]), 3), '|',
              'throttle :', round(float(abs(action[2])), 3), '|', 'direction :', round(float(action[3]), 3), '|',
              'total_reward :', round(episodic_reward, 6))

        car_controls.brake = 1 if float(action[0]) > 0.5 else 0
        car_controls.steering = float(action[1])
        car_controls.throttle = float(abs(action[2]))
        if action[3]:
            car_controls.manual_gear = 0
            car_controls.is_manual_gear = False
        else:
            car_controls.manual_gear = -1
            car_controls.is_manual_gear = True

        client.setCarControls(car_controls)

        # Recieve state and reward from environment.
        # state, reward, done, info = env.step(action)
        state = [client.getCarState().kinematics_estimated.position.x_val,  # 차량 위치 x 좌표
                 client.getCarState().kinematics_estimated.position.y_val,  # 차량 위치 y 좌표
                 client.getCarState().speed,                                # 차량 속도
                 client.getCarControls().brake,                             # 브레이크
                 client.getCarControls().steering,                          # 핸들 방향
                 client.getCarControls().throttle,                          # 차량 이동
                 client.getCarControls().manual_gear,                       # 후진 기어
                 client.getDistanceSensorData("Distance1").distance,        # 전방 거리 센서
                 client.getDistanceSensorData("Distance2").distance,        # 우측 거리 센서
                 client.getDistanceSensorData("Distance3").distance,        # 후방 거리 센서
                 client.getDistanceSensorData("Distance4").distance]        # 좌측 거리 센서

        # reward = 1/1000 if ((client.simGetCollisionInfo().object_name).lower()).find('pipesmall') >= 0 else -1

        if (((client.simGetCollisionInfo().object_name).lower()).find('pipesmall') >= 0):
            done = False
        else:
            print('Episode', ep, ': Crash!!')
            reward += -1
            done = True

        if (goal[0] > 0):
            if (6 < client.getCarState().kinematics_estimated.position.x_val < 8 and
                    goal[1] - 1 < client.getCarState().kinematics_estimated.position.y_val < goal[1] + 1):
                print('Episode', ep, ': Success!!')
                reward += 1
                done = True
        elif (goal[0] < 0):
            if (-9 < client.getCarState().kinematics_estimated.position.x_val < -7 and
                    goal[1] - 1 < client.getCarState().kinematics_estimated.position.y_val < goal[1] + 1):
                print('Episode', ep, ': Success!!')
                reward += 1
                done = True

        if prev_state[0] == state[0] and prev_state[1] == state[1]:
            reward += -1/1000

            if count == 0:
                count += 1
                start_time = time.time()
                end_time = time.time()
            else:
                count += 1
                end_time = time.time()

                if end_time - start_time >= 10:
                    print('Episode', ep, ': Don''t just stand there!!')
                    count = 0
                    reward += -1
                    done = True
        else:
            reward += 1/100000
            count = 0

        buffer.record((prev_state, action, reward, state))
        episodic_reward += reward

        buffer.learn()
        update_target(target_actor.variables, actor_model.variables, tau)
        update_target(target_critic.variables, critic_model.variables, tau)

        # End this episode when `done` is True
        if done:
            print('Final Reward :', episodic_reward)
            sim_stop()
            client, car_controls = sim_start()
            break

        prev_state = state

    ep_reward_list.append(episodic_reward)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)

# Plotting graph
# Episodes versus Avg. Rewards
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()

# Save the weights
actor_model.save_weights("parking_actor.h5")
critic_model.save_weights("parking_critic.h5")

target_actor.save_weights("parking_target_actor.h5")
target_critic.save_weights("parking_target_critic.h5")
