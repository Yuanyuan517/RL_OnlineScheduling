import numpy as np
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


class q_learning_funcs():

    def __init__(self, env, settings):
        self.env = env
        self.settings = settings
        self.epsilon = float(settings.get('Q_learning', 'epsilon'))
        self.discount_factor = float(settings.get('Q_learning', 'discount_factor'))
        self.alpha = float(settings.get('Q_learning', 'alpha'))
        self.num_episodes_train = int(settings.get('algorithms', 'num_episodes_train'))
        self.num_episodes_test = int(settings.get('algorithms', 'num_episodes_test'))
        self.size_time_steps = int(settings.get('algorithms', 'size_time_steps'))
        self.initial_seed = int(settings.get('algorithms', 'initial_seed'))
        self.episode_seeds = generate_random_seeds(self.initial_seed, self.num_episodes_test)

    # Make the $\epsilon$-greedy policy
    def create_epsilon_greedy_policy(self, Q):
        """
        Creates an epsilon-greedy policy based
        on a given Q-function and epsilon.

        Returns a function that takes the state
        as an input and returns the probabilities
        for each action in the form of a numpy array
        of length of the action space(set of possible actions).
        """
        def policy_function(state):
            # TO check: cz create just happens once, but action space varies, so i dont know if create_epsilon is necessary
            num_actions = state
            action_probabilities = np.ones(num_actions, dtype=float)*self.epsilon/num_actions
            # num of Q[state] can be greater than num_actions/state cz in each step, the situation can vary
            Q_values = Q[state][:num_actions]
            best_action = np.argmax(Q_values)
            #print("Action_prob before is ", action_probabilities)
            #print("Best_action is ", best_action)
            #print("QQ ", Q)
            action_probabilities[best_action] += (1.0-self.epsilon)
            #print("Action_prob after is ", action_probabilities)
            return action_probabilities

        return policy_function


    # Build Q-Learning Model
    def q_learning(self, plotting):
        """
        Q-Learning algorithm: Off-policy TD control.
        Finds the optimal greedy policy while improving
        following an epsilon-greedy policy"""

        # Keeps track of useful statistics

        stats = plotting.EpisodeStats(
            episode_lengths=np.zeros(self.num_episodes_train),
            episode_rewards=np.zeros(self.num_episodes_train))

        event_simu = EventSimulator2(self.settings)
        event_simu.set_randomness(True)
        total_tardiness = 0 # tardiness of all finished jobs

        # Action value function
        # A nested dictionary that maps
        # state -> (action -> action-value).
        Q = None
        granularity = 1
        # For every episode
        for i_episode in range(self.num_episodes_train):
            print("New Episode!!!!!!!!!!!!")
            total_tardiness = 0  # tardiness of all finished jobs
            max_tardinees = 0  # max tardiness among all finished + just-being-assigned jobs
            # Reset the environment and pick the first action
            env.state = self.env.reset(event_simu)
            if Q is None:
                Q = defaultdict(lambda: np.zeros(env.state))
              #  print("Q is ", Q)
                policy = self.create_epsilon_greedy_policy(Q)

            # Create an epsilon greedy policy function
            # appropriately for environment action space

            for t in range(self.size_time_steps):  # itertools.count():
                # Q = defaultdict(lambda: np.zeros(self.env.action_space.n))
                # Check decision epoch according to events
                # job release/job arrival (simulation strategy to be used?)
                # /machine idle
                # env.state[2] is machine list
                events = event_simu.event_simulation(t, env.machine, granularity)
                # update pt
                # released_new_jobs = events[1]
                # for new_job in released_new_job
                env.machine = events[2]
                tardiness = events[4]
               # print(" env waiting size ", len(env.waiting_jobs))
                if events[0]:
                    for job in events[1]:
                        env.waiting_jobs.append(job)
                env.state = len(env.waiting_jobs)
               # print(" new env waiting size ", len(env.waiting_jobs), "env state ", env.state)
                # env.remain_raw_pt -= events[3]

                # get probabilities of all actions from current state
                # if no released and waited job, then dummy action
                if env.state == 0 or env.machine.idle is False:
                    pass
                    # action = 0
                   # print("Action is 0")
                else:
                    action_probabilities = policy(env.state)
                   # print("Action prob is ", action_probabilities)

                    # choose action according to
                    # the probability distribution
                    action = np.random.choice(np.arange(
                        len(action_probabilities)),
                        p=action_probabilities)

                    # action may be over size
                    action = np.mod(action, env.state)
                   # print("Choose action ", action, " state ", env.state)

                    # take action and get reward, transit to next state
                    next_state, tardi, done, _ = self.env.step(action, events, t)

                    # Update statistics
                    # EDIT: April 20, 2020. use tardiness instead of reward
                    total_tardiness += tardiness
                    # stats.episode_rewards[i_episode] += reward
                    stats.episode_lengths[i_episode] = t

                    # April/21/2020: the reward takes into account total tardiness
                    # - tardiness of all finished jobs
                    # - prediction of the tardiness of the just selected job
                    reward = -1*(stats.episode_rewards[i_episode] + tardi)

                    # April 22, 2020-use max_tardinees to represent the result
                    max_tardinees = max_tardinees if tardi < max_tardinees else tardi
                    stats.episode_rewards[i_episode] = max_tardinees
                    #stats.episode_rewards[i_episode] = reward

                    # done is True if episode terminated
                    if done:
                        print("Episode finished")
                        break

                    # TD Update
                   # print("Test Q ", Q)
                   # print(" next state ", next_state)
                   # print(" Q[next_state] is ", Q[next_state], " env_state ", env.state)
                    if next_state >= len(Q[next_state]):
                        diff = next_state - len(Q[next_state]) + 1
                        for i in range(diff):
                            Q[next_state] = np.append(Q[next_state], 0)
                    best_next_action = np.argmax(Q[next_state])
                    td_target = reward + self.discount_factor * Q[next_state][best_next_action]
                    td_delta = td_target - Q[env.state][action]
                    Q[env.state][action] += self.alpha * td_delta
                  #  print("Now Q is ", Q)



                    env.state = next_state
                   # print("State updated to ", env.state)

        return Q, stats

    def q_learning_fixed_seed(self, Q, plotting):
        """
        Q-Learning algorithm: Off-policy TD control.
        Finds the optimal greedy policy while improving
        following an epsilon-greedy policy"""

        # Keeps track of useful statistics

        stats = plotting.EpisodeStats(
            episode_lengths=np.zeros(self.num_episodes_test),
            episode_rewards=np.zeros(self.num_episodes_test))

        event_simu = EventSimulator2(self.settings)
        event_simu.set_randomness(False)
        total_tardiness = 0 # tardiness of all finished jobs

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
            env.state = self.env.reset(event_simu)
            policy = self.create_epsilon_greedy_policy(Q)

            # differentiate seed for each episode
            seeds = generate_random_seeds(self.episode_seeds[i_episode], self.size_time_steps)

            for t in range(self.size_time_steps):  # itertools.count():
                # Q = defaultdict(lambda: np.zeros(self.env.action_space.n))
                # Check decision epoch according to events
                # job release/job arrival (simulation strategy to be used?)
                # /machine idle
                # env.state[2] is machine list
                event_simu.set_seed(seeds[t])
                events = event_simu.event_simulation(t, env.machine, granularity)
                # update pt
                # released_new_jobs = events[1]
                # for new_job in released_new_job
                env.machine = events[2]
                tardiness = events[4]
               # print(" env waiting size ", len(env.waiting_jobs))
                if events[0]:
                    for job in events[1]:
                        env.waiting_jobs.append(job)
                env.state = len(env.waiting_jobs)
               # print(" new env waiting size ", len(env.waiting_jobs), "env state ", env.state)
                # env.remain_raw_pt -= events[3]

                # get probabilities of all actions from current state
                # if no released and waited job, then dummy action
                if env.state == 0 or env.machine.idle is False:
                    pass
                    # action = 0
                    #print("Action is 0")
                else:
                    action_probabilities = policy(env.state)
                    #print("Action prob is ", action_probabilities)

                    # choose action according to
                    # the probability distribution
                    action = np.random.choice(np.arange(
                        len(action_probabilities)),
                        p=action_probabilities)

                    # action may be over size
                    action = np.mod(action, env.state)
                    #print("Choose action ", action, " state ", env.state)

                    # take action and get reward, transit to next state
                    next_state, tardi, done, _ = self.env.step(action, events, t)

                    # Update statistics
                    # EDIT: April 20, 2020. use tardiness instead of reward
                    total_tardiness += tardiness
                    # stats.episode_rewards[i_episode] += reward
                    stats.episode_lengths[i_episode] = t

                    # April/21/2020: the reward takes into account total tardiness
                    # - tardiness of all finished jobs
                    # - prediction of the tardiness of the just selected job
                    reward = -1*(stats.episode_rewards[i_episode] + tardi)

                    # April 22, 2020-use max_tardinees to represent the result
                    max_tardinees = max_tardinees if tardi < max_tardinees else tardi
                    stats.episode_rewards[i_episode] = max_tardinees
                    #stats.episode_rewards[i_episode] = reward

                    # done is True if episode terminated
                    if done:
                        break

                    # TD Update
                   # print("Test Q ", Q)
                    #print(" next state ", next_state)
                   # print(" Q[next_state] is ", Q[next_state], " env_state ", env.state)
                    if next_state >= len(Q[next_state]):
                        diff = next_state - len(Q[next_state]) + 1
                        for i in range(diff):
                            Q[next_state] = np.append(Q[next_state], 0)
                    best_next_action = np.argmax(Q[next_state])
                    td_target = reward + self.discount_factor * Q[next_state][best_next_action]
                    td_delta = td_target - Q[env.state][action]
                    Q[env.state][action] += self.alpha * td_delta
                   # print("Now Q is ", Q)
                    env.state = next_state
                   # print("State updated to ", env.state)

        return Q, stats


if __name__ == '__main__':
    matplotlib.style.use('ggplot')
    plotting = Plotting()
    env = JSPEnv2()
    _conf = ConfigParser()
    _conf.read('./etc/app.ini')
    # num_episode = 500
    #  Train the model
    Q_learn = q_learning_funcs(env, _conf)
    # event simulator is not fixed
    Q, stats = Q_learn.q_learning(plotting)
    print("stats ", stats)
    plotting.plot_episode_stats(stats)

    # event simulator is fixed
    # test the model with calculated Q
    # Q_learn.num_episodes = 10
    Q2, stats2 = Q_learn.q_learning_fixed_seed(Q, plotting)
    print("New Stats", stats2)
    plotting.plot_episode_stats(stats2)
