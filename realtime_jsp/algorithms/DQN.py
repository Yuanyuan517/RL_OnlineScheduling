from random import sample

import numpy as np
import random_self
#from IPython.display import clear_output
from collections import deque
import progressbar
import matplotlib.style

import gym

from tensorflow.keras import models, Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam
from environments.JSPEnv2 import JSPEnv2
from configparser import ConfigParser
from simulators.eventsimulator2 import EventSimulator2
from utilities.plotting import Plotting
from simulators.utility import generate_random_seeds

# https://rubikscode.net/2019/07/08/deep-q-learning-with-python-and-tensorflow-2-0/

class dqn:
    def __init__(self, env, optimizer, settings):
        # Initialize atributes
        self.settings = settings
        self.env = env
        self._state_size = 2000#15#enviroment.observation_space.n
        self._action_size = 2000#10#enviroment.action_space.n
        self._optimizer = optimizer

        self.expirience_replay = deque(maxlen=2000)

        # Initialize discount and exploration rate
        self.gamma = float(settings.get('Q_learning', 'discount_factor'))##0.6
        self.epsilon = float(settings.get('Q_learning', 'epsilon'))#0.1

        # Build networks
        # 1. Q-Network calculates Q-Value in the state St
        self.q_network = self._build_compile_model()
        # 2. Target-Network calculates Q-Value in the state St+1
        self.target_network = self._build_compile_model()
        self.alighn_target_model()

        self.num_episodes_trains = settings.get('algorithms', 'num_episodes_trains').split()
        self.num_episodes_train = 0
        self.num_episodes_test = int(settings.get('algorithms', 'num_episodes_test'))
        self.size_time_steps = 5000#int(settings.get('algorithms', 'size_time_steps'))
        self.initial_seed = int(settings.get('algorithms', 'initial_seed'))
        self.episode_seeds = generate_random_seeds(self.initial_seed, self.num_episodes_test)

        self.criteria = 2  # 1 is only DD, 2 DD+pt, 3 random
        self.obj = 2  # 1 is min max tardiness, 2 is min total tardiness
        self.name = "DQN"
        self.save_path = "dqnV2.h5"

    # the agent has to store previous experiences in a local memory
    def store(self, state, action, reward, next_state, terminated):
        self.expirience_replay.append((state, action, reward , next_state, terminated))

    def _build_compile_model(self):
        model = Sequential()
        #model.add(Embedding(self._state_size, 10, input_length=1)) # may not be necessary, exclude later
        #model.add(Reshape((10,)))
        model.add(Dense(12, input_shape=(1,), activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(50, activation='relu'))
        # the action size is not fixed, i can use mod, but is it right way to do???
        model.add(Dense(self._action_size, activation='linear'))

        model.compile(loss='mse', optimizer=self._optimizer)
        return model

    # update target network
    def alighn_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())

    # return predicted action from the trained model
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # pick a random action
            return self.env.action_space.sample()
        # invoke Q network to make a prediction
        array = self.convert_2_dim(state)
        q_values = self.q_network.predict(array)
        return np.argmax(q_values[0])

    def takeDueTime(self, job):
        if self.criteria == 1:
            return job.due_t
        if self.criteria == 2:
            return job.due_t + job.pt

    def convert_2_dim(self, state):
        array = np.array(state)
        array = np.append(array, state)  # use 1 d there is error... TO CHECK !!!!!!!!!!!!
        return array

    # pick random samples from the experience replay memory and train the Q-Network
    def retrain(self, batch_size):
        minibatch = sample(self.expirience_replay, batch_size)

        for state, action, reward, next_state, terminated in minibatch:
            array1 = self.convert_2_dim(state)
            target = self.q_network.predict(array1)

            if terminated:
                target[0][action] = reward
            else:
                array2 = self.convert_2_dim(next_state)
                t = self.target_network.predict(array2)
                target[0][action] = reward + self.gamma * np.amax(t)

            # self.q_network.fit(state, target, epochs=1, verbose=0)
            self.q_network.fit(array1, target, epochs=1, verbose=0)

    def training_steps(self):
        event_simu = EventSimulator2(self.settings)
        event_simu.set_randomness(True)
        granularity = 1
        for e in range(0, self.num_episodes_train):
            # Reset the environment
            # Reset the environment and pick the first action
            self.env.state = self.env.reset(event_simu)
            self.env.state = np.reshape(self.env.state, [1])
            total_tardiness = 0  # tardiness of all finished jobs
            #state = self.enviroment.reset()
            #state = np.reshape(state, [1, 1])

            # Initialize variables
            # reward = 0
            terminated = False

            bar = progressbar.ProgressBar(maxval=self.size_time_steps / 10,
                                          widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar.start()

            for timestep in range(self.size_time_steps):
                # Run Action
                action = self.act(self.env.state)

                # environment related ops
                events = event_simu.event_simulation(timestep, self.env.machine, granularity)

                #self.env.machine = events[2]
                #tardiness = events[4]
                # print(" env waiting size ", len(env.waiting_jobs))
                if events[0]:
                    for job in events[1]:
                        self.env.waiting_jobs.append(job)
                if self.criteria != 3:
                    self.env.waiting_jobs.sort(key=self.takeDueTime)
                if self.env.state == 0: # no job is waiting
                    pass
                else:
                    #print("Ch1 ", action, self.env.state)
                    if type(self.env.state) is int:
                        action = action % self.env.state
                    else:
                        action = action % self.env.state[0]
                    #print("Ch2 ", action)
                    # Take action
                    next_state, tardi, done, updated_machine = self.env.step(action, event_simu, timestep, granularity)
                    self.env.machine = updated_machine
                    # Update statistics
                    total_tardiness += tardi

                    # April/21/2020: the reward takes into account total tardiness
                    # - tardiness of all finished jobs
                    reward = -1 * total_tardiness
                    agent.store(self.env.state, action, reward, next_state, terminated)

                    self.env.state = next_state

                if terminated:
                    self.alighn_target_model()
                    break

                if len(agent.expirience_replay) > batch_size:
                    self.retrain(batch_size)

                if timestep % 10 == 0:
                    bar.update(timestep / 10 + 1)

            bar.finish()
            if (e + 1) % 10 == 0:
                print("**********************************")
                print("Episode: {}".format(e + 1))
                # enviroment.render()
                print("**********************************")

        # save network, https://www.tensorflow.org/tutorials/keras/save_and_load
        self.q_network.save(self.save_path)
        print('Successfully saved: ' + self.save_path)

    # https://github.com/ShanHaoYu/Deep-Q-Network-Breakout/blob/master/test.py
    def test(self, plotting):
        # Keeps track of useful statistics
        stats = plotting.EpisodeStats(
            episode_lengths=np.zeros(self.num_episodes_test),
            episode_rewards=np.zeros(self.num_episodes_test),
            episode_obj=np.zeros(self.num_episodes_test),
        )

        event_simu = EventSimulator2(self.settings)
        event_simu.set_randomness(False)

        # Action value function
        # A nested dictionary that maps
        # state -> (action -> action-value).
        granularity = 1
        # For every episode
        for i_episode in range(self.num_episodes_test):
            print("New Episode!!!!!!!!!!!! ", i_episode)
            total_tardiness = 0  # tardiness of all finished jobs
            max_tardinees = 0  # max tardiness among all finished + just-being-assigned jobs
            # Reset the environment and pick the first action
            self.env.state = self.env.reset(event_simu)

            # differentiate seed for each episode
            seeds = generate_random_seeds(self.episode_seeds[i_episode], self.size_time_steps)

            for t in range(self.size_time_steps):  # itertools.count():
                # Q = defaultdict(lambda: np.zeros(self.env.action_space.n))
                # Check decision epoch according to events
                # job release/job arrival (simulation strategy to be used?)
                # /machine idle
                # env.state[2] is machine list
                event_simu.set_seed(seeds[t])
                events = event_simu.event_simulation(t, self.env.machine, granularity)
                # update pt
                # released_new_jobs = events[1]
                # for new_job in released_new_job
                # self.env.machine = events[2]
                # tardiness = events[4]
                # print(" env waiting size ", len(env.waiting_jobs))
                if events[0]:
                    for job in events[1]:
                        self.env.waiting_jobs.append(job)
                # sort jobs according to the due date+pt?, 1st one is the one with smallest due date (urgent)
                if self.criteria != 3:
                    self.env.waiting_jobs.sort(key=self.takeDueTime)
                self.env.state = len(self.env.waiting_jobs)
                # print(" new env waiting size ", len(env.waiting_jobs), "env state ", env.state)
                # env.remain_raw_pt -= events[3]

                # get probabilities of all actions from current state
                # if no released and waited job, then dummy action
                if self.env.state == 0:  # or env.machine.idle is False:
                    pass
                    # action = 0
                    # print("Action is 0")
                else:
                    # choose action from the prediction of q network
                    action = self.act(self.env.state)

                    # action may be over size
                    action = np.mod(action, self.env.state)
                    # print("Choose action ", action, " state ", self.env.state)

                    # take action and get reward, transit to next state
                    next_state, tardi, done, updated_machine = self.env.step(action, event_simu, t, granularity)
                    self.env.machine = updated_machine
                    # Update statistics
                    total_tardiness += tardi
                    # stats.episode_rewards[i_episode] += reward
                    stats.episode_lengths[i_episode] = t

                    # April/21/2020: the reward takes into account total tardiness
                    # - tardiness of all finished jobs
                    reward = -1 * total_tardiness
                    stats.episode_rewards[i_episode] += reward

                    # April 22, 2020-use max_tardinees to represent the result
                    max_tardinees = max_tardinees if tardi < max_tardinees else tardi
                    # April 26: enable the option of min total tardiness
                    if self.obj == 1:
                        stats.episode_obj[i_episode] = max_tardinees # note, here the reward is not the actual reward but the obj for comparing performance
                    else:
                        stats.episode_obj[i_episode] = total_tardiness
                    # done is True if episode terminated
                    if done:
                        break

                    '''
                    # TD Update
                    if next_state >= len(Q[next_state]):
                        diff = next_state - len(Q[next_state]) + 1
                        for i in range(diff):
                            Q[next_state] = np.append(Q[next_state], 0)
                    best_next_action = np.argmax(Q[next_state])
                    td_target = reward + self.discount_factor * Q[next_state][best_next_action]
                    td_delta = td_target - Q[self.env.state][action]
                    Q[self.env.state][action] += self.alpha * td_delta
                    '''
                    self.env.state = next_state
                # print("State updated to ", env.state)

        return stats


if __name__ == '__main__':
    optimizer = Adam(learning_rate=0.01)
    env = JSPEnv2()
    _conf = ConfigParser()
    _conf.read('/Users/yuanyuanli/PycharmProjects/RL-RealtimeScheduling/realtime_jsp'
                     '/etc/app.ini')

    batch_size = 32
    #num_of_episodes = 100
    #timesteps_per_episode = 1000

    agent = dqn(env, optimizer, _conf)
    # load saved model
    agent.q_network = models.load_model(agent.save_path)
    agent.q_network.summary()
    # train
    #for num in agent.num_episodes_trains:
    #    agent.num_episodes_train = int(num)
    #    agent.training_steps()
    # test
    matplotlib.style.use('ggplot')
    plotting = Plotting()
    stats2 = agent.test(plotting)
    plotting.plot_episode_stats(stats2)
    print("New Stats", stats2)
