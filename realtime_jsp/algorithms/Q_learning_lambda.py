import numpy as np
import itertools
import matplotlib.style
from collections import defaultdict
from simulators.eventsimulator2 import EventSimulator2
from configparser import ConfigParser
from environments.JSPEnv2 import JSPEnv2
from utilities.plotting import Plotting
from simulators.utility import generate_random_seeds


'''
Based on: https://www.geeksforgeeks.org/q-learning-in-python/
'''


class Q_Lambda():

    def __init__(self, env, settings):
        self.env = env
        self.settings = settings
        self.lamb = float(settings.get('Q_learning', 'lamb'))
        self.epsilon = float(settings.get('Q_learning', 'epsilon'))
        self.discount_factor = float(settings.get('Q_learning', 'discount_factor'))
        self.alpha = float(settings.get('Q_learning', 'alpha'))
        self.num_episodes_trains = settings.get('algorithms', 'num_episodes_trains').split()
        self.num_episodes_train = 500#int(settings.get('algorithms', 'num_episodes_train'))
        self.num_episodes_test = 50#int(settings.get('algorithms', 'num_episodes_test'))
        self.size_time_steps = int(settings.get('algorithms', 'size_time_steps'))
        self.initial_seed = int(settings.get('algorithms', 'initial_seed'))
        self.episode_seeds = generate_random_seeds(self.initial_seed, self.num_episodes_test)
        # calculate number of actions and states
        self.criterion = [1, 3]#[1, 2]#, 3]
        self.criteria = 1  # 1 is only DD, 2 DD+pt, 3 random
        self.obj = 2  # 1 is min max tardiness, 2 is min total tardiness
        self.name = "QLamb"
        self.state_size_max = 2000 # an estimated value from the past experiment
        self.action_size_max = 2000

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
            action_probabilities[best_action] += (1.0-self.epsilon)
            return action_probabilities

        return policy_function

    def takeDueTime(self, job):
        if self.criteria == 1:
            return job.due_t
        if self.criteria == 2:
            return job.due_t + job.pt

    # Build Q-Learning Model
    def learn(self, plotting):
        """
        Q-Learning algorithm: Off-policy TD control.
        Finds the optimal greedy policy while improving
        following an epsilon-greedy policy"""

        # Keeps track of useful statistics

        stats = plotting.EpisodeStats(
            episode_lengths=np.zeros(self.num_episodes_train),
            episode_rewards=np.zeros(self.num_episodes_train),
            episode_obj=np.zeros(self.num_episodes_train)
        )

        event_simu = EventSimulator2(self.settings)
        event_simu.set_randomness(True)

        # Action value function
        # A nested dictionary that maps
        # state -> (action -> action-value).
        Q =np.zeros((self.state_size_max, self.action_size_max))#None
        policy = self.create_epsilon_greedy_policy(Q)
        granularity = 1
        # For every episode
        for i_episode in range(self.num_episodes_train):
            print("TrainNew Episode ", i_episode)
            total_tardiness = 0  # tardiness of all finished jobs
            max_tardinees = 0  # max tardiness among all finished + just-being-assigned jobs
            # Reset the environment and pick the first action
            self.env.state = self.env.reset(event_simu)
            '''
            if Q is None:
                Q = defaultdict(lambda: np.zeros(self.env.state))
                policy = self.create_epsilon_greedy_policy(Q)
            '''
            eligibility = np.zeros((self.state_size_max, self.action_size_max))

            for t in range(self.size_time_steps):
                events = event_simu.event_simulation(t, self.env.machine, granularity)

                if events[0]:
                    for job in events[1]:
                        self.env.waiting_jobs.append(job)
                # sort jobs according to the due date+pt?, 1st one is the one with smallest due date (urgent)
                if self.criteria != 3:
                    self.env.waiting_jobs.sort(key=self.takeDueTime)

                # get probabilities of all actions from current state
                # if no released and waited job, then dummy action
                if self.env.state == 0: #or env.machine.idle is False:
                    pass
                else:
                    if t==0:
                        action_probabilities = policy(self.env.state)

                        # choose action according to
                        # the probability distribution
                        action = np.random.choice(np.arange(
                            len(action_probabilities)),
                            p=action_probabilities)

                    # action may be over size
                    action = np.mod(action, self.env.state)

                    # take action and get reward, transit to next state
                    next_state, tardi, done, updated_machine = self.env.step(action, event_simu, t, granularity)
                    self.env.machine = updated_machine
                    # Update statistics
                    total_tardiness += tardi
                    stats.episode_lengths[i_episode] = t

                    # April/21/2020: the reward takes into account total tardiness
                    # - tardiness of all finished jobs
                    reward = -1*total_tardiness
                    stats.episode_rewards[i_episode] += reward

                    '''
                    # April 22, 2020-use max_tardinees to represent the result
                    max_tardinees = max_tardinees if tardi < max_tardinees else tardi
                    # April 26: enable the option of min total tardiness
                    if self.obj == 1:
                        stats.episode_obj[i_episode] = max_tardinees
                    else:
                    '''
                    stats.episode_obj[i_episode] = total_tardiness

                    # done is True if episode terminated
                    if done:
                        #print("Episode finished")
                        break

                    '''
                    if next_state >= len(Q[next_state]):
                        diff = next_state - len(Q[next_state]) + 1
                        for i in range(diff):
                            Q[next_state] = np.append(Q[next_state], 0)
                    '''
                    # update eligibility
                    eligibility[self.env.state][action] += 1.0
                    # TD Update
                    best_next_action = np.argmax(Q[next_state])
                    td_target = reward + self.discount_factor * Q[next_state][best_next_action]
                    td_delta = td_target - Q[self.env.state][action]
                    '''
                    for i in range(len(Q)):
                        for j in range(len(Q[i])):
                    '''
                    Q += self.alpha * td_delta * eligibility

                    action_probabilities = policy(self.env.state)

                    # choose action according to
                    # the probability distribution
                    next_action = np.random.choice(np.arange(
                        len(action_probabilities)),
                        p=action_probabilities)
                    # https://stackoverflow.com/questions/40862578/how-to-understand-watkinss-q%CE%BB-learning-algorithm-in-suttonbartos-rl-book
                    if next_action == best_next_action:
                        eligibility *= self.lamb*self.discount_factor
                    else:
                        eligibility = np.zeros((self.state_size_max, self.action_size_max))

                    self.env.state = next_state
                    action = next_action

        return Q, stats

    def fixed_seed(self, Q, plotting):
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

        # Action value function
        # A nested dictionary that maps
        # state -> (action -> action-value).
        granularity = 1
        # For every episode
        for i_episode in range(self.num_episodes_test):
            #print("New Episode!!!!!!!!!!!! ", i_episode)
            total_tardiness = 0  # tardiness of all finished jobs
            max_tardinees = 0  # max tardiness among all finished + just-being-assigned jobs
            # Reset the environment and pick the first action
            self.env.state = self.env.reset(event_simu)
            policy = self.create_epsilon_greedy_policy(Q)

            # differentiate seed for each episode
            seeds = generate_random_seeds(self.episode_seeds[i_episode], self.size_time_steps)

            eligibility = np.zeros((self.state_size_max, self.action_size_max))

            for t in range(self.size_time_steps):  # itertools.count():
                # Q = defaultdict(lambda: np.zeros(self.env.action_space.n))
                # Check decision epoch according to events
                # job release/job arrival (simulation strategy to be used?)
                # /machine idle
                # env.state[2] is machine list
                event_simu.set_seed(seeds[t])
                event_simu.episode = i_episode
                events = event_simu.event_simulation(t, self.env.machine, granularity)
                if events[0]:
                    for job in events[1]:
                        self.env.waiting_jobs.append(job)
                # sort jobs according to the due date+pt?, 1st one is the one with smallest due date (urgent)
                if self.criteria != 3:
                    self.env.waiting_jobs.sort(key=self.takeDueTime)
                self.env.state = len(self.env.waiting_jobs)

                # get probabilities of all actions from current state
                # if no released and waited job, then dummy action
                if self.env.state == 0: #or env.machine.idle is False:
                    pass
                else:
                    if t==0:
                        action_probabilities = policy(self.env.state)

                        # choose action according to
                        # the probability distribution
                        action = np.random.choice(np.arange(
                            len(action_probabilities)),
                            p=action_probabilities)

                    # action may be over size
                    action = np.mod(action, self.env.state)

                    # June 16, 2020: for debugging
                    self.env.episode = i_episode
                    # take action and get reward, transit to next state
                    next_state, tardi, done, updated_machine = self.env.step(action, event_simu, t, granularity)
                    self.env.machine = updated_machine
                    # Update statistics
                    total_tardiness += tardi
                    stats.episode_lengths[i_episode] = t

                    # April/21/2020: the reward takes into account total tardiness
                    # - tardiness of all finished jobs
                    reward = -1*total_tardiness
                    stats.episode_rewards[i_episode] += reward

                    '''
                    # April 22, 2020-use max_tardinees to represent the result
                    max_tardinees = max_tardinees if tardi < max_tardinees else tardi
                    # April 26: enable the option of min total tardiness
                    if self.obj == 1:
                        stats.episode_obj[i_episode] = max_tardinees
                    else:
                    '''
                    stats.episode_obj[i_episode] = total_tardiness
                    if i_episode == 35:
                        print("Total tardiness is ", total_tardiness, " Current tard is ", tardi)

                    # done is True if episode terminated
                    if done:
                        break

                    '''
                    if next_state >= len(Q[next_state]):
                        diff = next_state - len(Q[next_state]) + 1
                        for i in range(diff):
                            Q[next_state] = np.append(Q[next_state], 0)
                    '''
                    # update eligibility
                    eligibility[self.env.state][action] += 1.0

                    # TD Update
                    best_next_action = np.argmax(Q[next_state])
                    td_target = reward + self.discount_factor * Q[next_state][best_next_action]
                    td_delta = td_target - Q[self.env.state][action]
                    '''
                     for i in range(len(Q)):
                        for j in range(len(Q[i])):
                    '''
                    Q += self.alpha * td_delta * eligibility

                    action_probabilities = policy(self.env.state)

                    # choose action according to
                    # the probability distribution
                    next_action = np.random.choice(np.arange(
                        len(action_probabilities)),
                        p=action_probabilities)
                    # https://stackoverflow.com/questions/40862578/how-to-understand-watkinss-q%CE%BB-learning-algorithm-in-suttonbartos-rl-book
                    if next_action == best_next_action:
                        eligibility *= self.lamb * self.discount_factor
                    else:
                        eligibility = np.zeros((self.state_size_max, self.action_size_max))

                    self.env.state = next_state
                    action = next_action

        return Q, stats


if __name__ == '__main__':
    matplotlib.style.use('ggplot')
    plotting = Plotting()
    env = JSPEnv2()
    _conf = ConfigParser()
    _conf.read('/Users/yuanyuanli/PycharmProjects/RL-RealtimeScheduling/realtime_jsp'
                     '/etc/app.ini')
    # num_episode = 500
    #  Train the model
    Q_L = Q_Lambda(env, _conf)
    filename = '/Users/yuanyuanli/PycharmProjects/RL-RealtimeScheduling/realtime_jsp/results/' \
               'QL_lateness_2500.txt'
               # 'QLV3_2500.txt'
               #'bestSettingQL.txt'
    # '1000V3.txt'
    time = [2500] #100]#, 5000]
    with open(filename, 'a') as f:
        for t in time:
            Q_L.size_time_steps = t
            for c in Q_L.criterion:
                Q_L.criteria = c
                # event simulator is not fixed
                Q, stats = Q_L.learn(plotting)
                # print("stats ", stats)
                #plotting.plot_episode_stats(stats)

                # event simulator is fixed
                # test the model with calculated Q
                # Q_learn.num_episodes = 10
                Q2, stats2 = Q_L.fixed_seed(Q, plotting)
                print("New Stats", stats2)
                cri = ""
                if Q_L.criteria == 1:
                    cri = "DD"
                elif Q_L.criteria == 2:
                    cri = "DD_pt"
                else:
                    cri = "random"
                s = "OL " + str(t) + " " + cri #+" "+str(lambda_value)
                f.write(s)
                f.write(" ")
                b = np.matrix(stats2.episode_obj)
                np.savetxt(f, b, fmt="%d")
                f.write("\n")
                #plotting.plot_episode_stats(stats2)
    print("Finished QL")
