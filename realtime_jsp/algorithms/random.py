import numpy as np
import random
import itertools
import matplotlib.style
from collections import defaultdict
from realtime_jsp.simulators.eventsimulator2 import EventSimulator2
from configparser import ConfigParser
from realtime_jsp.environments.JSPEnv2 import JSPEnv2
from realtime_jsp.utilities.plotting import Plotting
from realtime_jsp.simulators.utility import generate_random_seeds

'''
Based on: https://www.geeksforgeeks.org/q-learning-in-python/
'''


class Random():

    def __init__(self, env, settings):
        self.env = env
        self.settings = settings
        self.num_episodes_test = int(settings.get('algorithms', 'num_episodes_test'))
        self.size_time_steps = int(settings.get('algorithms', 'size_time_steps'))
        self.initial_seed = int(settings.get('algorithms', 'initial_seed'))
        self.episode_seeds = generate_random_seeds(self.initial_seed, self.num_episodes_test)
        self.action_set = set()
        self.state_set = set()
        self.obj = 2  # 1 is min max tardiness, 2 is min total tardiness
        self.name = "Random"

    def run(self, plotting):
        """
        Q-Learning algorithm: Off-policy TD control.
        Finds the optimal greedy policy while improving
        following an epsilon-greedy policy"""

        # Keeps track of useful statistics

        stats = plotting.EpisodeStats(
            episode_lengths=np.zeros(self.num_episodes_test),
            episode_rewards=np.zeros(self.num_episodes_test),
            episode_obj=np.zeros(self.num_episodes_test))

        event_simu = EventSimulator2(self.settings)
        event_simu.set_randomness(False)

        granularity = 1
        # For every episode
        for i_episode in range(self.num_episodes_test):
            #print("New Episode!!!!!!!!!!!! ", i_episode)
            total_tardiness = 0  # tardiness of all finished jobs
            max_tardinees = 0  # max tardiness among all finished + just-being-assigned jobs

            # Reset the environment and pick the first action
            self.env.state = self.env.reset(event_simu)
            self.state_set.add(self.env.state)
            # Create an epsilon greedy policy function
            # appropriately for environment action space

            # differentiate seed for each episode
            seeds = generate_random_seeds(self.episode_seeds[i_episode], self.size_time_steps)

            for t in range(self.size_time_steps):  # itertools.count():
                # Q = defaultdict(lambda: np.zeros(self.env.action_space.n))
                # Check decision epoch according to events
                # job release/job arrival (simulation strategy to be used?)
                # /machine idle
                # env.state[2] is machine list
                # set fixed seed for testing
                event_simu.set_seed(seeds[t])
                events = event_simu.event_simulation(t, self.env.machine, granularity)
                # update pt
                # released_new_jobs = events[1]
                # for new_job in released_new_job
                #self.env.machine = events[2]
                #tardiness = events[4]
               #print(" env waiting size ", len(env.waiting_jobs))
                if events[0]:
                    for job in events[1]:
                        self.env.waiting_jobs.append(job)
                self.env.state = len(self.env.waiting_jobs)
                self.state_set.add(self.env.state)
                #print(" new env waiting size ", len(env.waiting_jobs), "env state ", env.state)
                # env.remain_raw_pt -= events[3]

                # EDIT-23/04/2020: enable preemption
                if self.env.state == 0: #or env.machine.idle is False:
                    pass
                else:
                    # choose action randomly
                    action = random.randint(1, self.env.state) - 1
                    self.action_set.add(action)
                    # print("Choose action ", action, " state ", env.state)

                    # take action and get reward, transit to next state
                    self.env.debug_waiting_jobs()
                    next_state, tardi, done, updated_machine = self.env.step(action, event_simu, t, granularity)
                    self.env.machine = updated_machine
                    # Update statistics
                    total_tardiness += tardi
                    # stats.episode_rewards[i_episode] += reward
                    stats.episode_lengths[i_episode] = t

                    # April 22, 2020-use max_tardinees to represent the result
                    max_tardinees = max_tardinees if tardi < max_tardinees else tardi
                    if self.obj == 1:
                        stats.episode_obj[i_episode] = max_tardinees
                    else:
                        stats.episode_obj[i_episode] = total_tardiness
                    #stats.episode_rewards[i_episode] = reward

                    # done is True if episode terminated
                    if done:
                        print("Episode finished")
                        #print("state size ", len(random_model.state_set), random_model.state_set)
                        #print("action size ", len(random_model.action_set), random_model.action_set)
                        break

                    self.env.state = next_state
                   # print("State updated to ", env.state)

        return stats


if __name__ == '__main__':
    matplotlib.style.use('ggplot')
    plotting = Plotting()
    env = JSPEnv2()
    _conf = ConfigParser()
    _conf.read('/Users/yuanyuanli/PycharmProjects/RL-RealtimeScheduling/realtime_jsp'
                     '/etc/app.ini')
    #  Test
    random_model = Random(env, _conf)
    stats = random_model.run(plotting)
    print("stats ", stats)
    plotting.plot_episode_stats(stats)
    print("state size ", len(random_model.state_set), random_model.state_set)
    print("action size ", len(random_model.action_set), random_model.action_set)
