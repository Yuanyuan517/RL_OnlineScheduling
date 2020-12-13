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
Based on: https://www.geeksforgeeks.org/sarsa-reinforcement-learning/
'''

class SARSA_L():

    def __init__(self, env, settings):
        self.env = env
        self.settings = settings
        self.lamb = float(settings.get('Q_learning', 'lamb'))
        self.epsilon = float(settings.get('Q_learning', 'epsilon'))
        self.discount_factor = float(settings.get('Q_learning', 'discount_factor'))
        self.alpha = float(settings.get('Q_learning', 'alpha'))
        self.num_episodes_trains = settings.get('algorithms', 'num_episodes_trains').split()
        self.num_episodes_train = 500  # int(settings.get('algorithms', 'num_episodes_train'))
        self.num_episodes_test = 50  # int(settings.get('algorithms', 'num_episodes_test'))
        self.size_time_steps = int(settings.get('algorithms', 'size_time_steps'))
        self.initial_seed = int(settings.get('algorithms', 'initial_seed'))
        self.episode_seeds = generate_random_seeds(self.initial_seed, self.num_episodes_test)
        # calculate number of actions and states
        self.criterion = [1, 2, 3]
        self.criteria = 1 # 1 is only DD, 2 DD+pt, 3 random
        self.obj = 2 # 1 is min max tardiness, 2 is min total tardiness
        self.name = "SarsaLambda"
        self.state_size_max = 2000  # an estimated value from the past experiment
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
            num_actions = state
            action_probabilities = np.ones(num_actions, dtype=float) * self.epsilon / num_actions
            # num of Q[state] can be greater than num_actions/state cz in each step, the situation can vary
            #print("Q ", Q)
            Q_values = Q[state][:num_actions]
            best_action = np.argmax(Q_values)
            #print("Action_prob before is ", action_probabilities)
            #print("Best_action is ", best_action)
            #print("QQ ", Q)
            action_probabilities[best_action] += (1.0 - self.epsilon)
            #print("Action_prob after is ", action_probabilities)
            return action_probabilities

        return policy_function

    # TO CHECK: possible and only difference between SARSA and Q learning
    # Function to choose the next action
    def choose_action(self, Q, state):
        num_actions = state
        if np.random.uniform(0, 1) < self.epsilon:
            actions = np.random.randint(0, num_actions, 1)
            action = actions[0]
        else:
            Q_values = Q[state][:num_actions]
            action = np.argmax(Q_values)
        return action

    def takeDueTime(self, job):
        if self.criteria == 1:
            return job.due_t
        if self.criteria == 2:
            return job.due_t  +job.pt

        # Build sarsa  Model
    def learn(self, plotting):
        """
        Q-Learning algorithm: Off-policy TD control.
        Finds the optimal greedy policy while improving
        following an epsilon-greedy policy"""

        # Keeps track of useful statistics

        stats = plotting.EpisodeStats(
            episode_lengths=np.zeros(self.num_episodes_train),
            episode_rewards=np.zeros(self.num_episodes_train),
            episode_obj=np.zeros(self.num_episodes_train))

        event_simu = EventSimulator2(self.settings)
        total_tardiness = 0  # tardiness of all finished jobs
        event_simu.set_randomness(True)

        # Action value function
        # A nested dictionary that maps
        # state -> (action -> action-value).
        Q = np.zeros((self.state_size_max, self.action_size_max))#None
        policy = self.create_epsilon_greedy_policy(Q)
        granularity = 1
        # For every episode
        for i_episode in range(self.num_episodes_train):
            print("TrainNew Episode ", i_episode)
            total_tardiness = 0  # tardiness of all finished jobs
            max_tardiness = 0  # max tardiness among all finished + just-being-assigned jobs
            # Reset the environment and pick the first action
            self.env.state = self.env.reset(event_simu)
            '''
            if Q is None:
                Q = defaultdict(lambda: np.zeros(self.env.state))
                #print("Q is ", Q)
                policy = self.create_epsilon_greedy_policy(Q)
            '''

            eligibility = np.zeros((self.state_size_max, self.action_size_max))

            for t in range(self.size_time_steps):
                events = event_simu.event_simulation(t, self.env.machine, granularity)
                if events[0]:
                    for job in events[1]:
                        self.env.waiting_jobs.append(job)
                self.env.state = len(self.env.waiting_jobs)
                # sort jobs according to the due date+pt?, 1st one is the one with smallest due date (urgent)
                if self.criteria != 3:
                    self.env.waiting_jobs.sort(key=self.takeDueTime)

                # if no released and waited job, then dummy action
                if self.env.state == 0: # or env.machine.idle is False:
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
                    # print("Choose action ", action, " state ", env.state)

                    # take action and get reward, transit to next state
                    next_state, tardi, done, updated_machine = self.env.step(action, event_simu, t, granularity)
                    self.env.machine = updated_machine
                    # Update statistics
                    # EDIT: April 20, 2020. use tardiness instead of reward
                    total_tardiness += tardi
                    # stats.episode_rewards[i_episode] += reward
                    stats.episode_lengths[i_episode] = t

                    # April/21/2020: the reward takes into account total tardiness
                    # - tardiness of all finished jobs
                    reward = -1*total_tardiness
                    stats.episode_rewards[i_episode] += reward

                    # April 22, 2020-use max_tardiness to represent the result
                    max_tardiness = max_tardiness if tardi < max_tardiness else tardi
                    # April 26: enable the option of min total tardiness
                    #if self.obj == 1:
                    #    stats.episode_obj[i_episode] = max_tardiness
                    #else:
                    stats.episode_obj[i_episode] = total_tardiness
                    # print("Tardi is ", tardi, " max tardi is ", max_tardiness)

                    # done is True if episode terminated
                    if done:
                        #print("All jobs finished")
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
                    best_next_action = self.choose_action(Q, next_state)# np.argmax(Q[next_state])
                    td_target = reward + self.discount_factor * Q[next_state][best_next_action]
                    td_delta = td_target - Q[self.env.state][action]
                    # for i in range(len(Q)):
                    #    for j in range(len(Q[i])):
                    Q += self.alpha * td_delta * eligibility

                    self.env.state = next_state
                    action = best_next_action

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
            print("New Episode!!!!!!!!!!!! ", i_episode)
            total_tardiness = 0  # tardiness of all finished jobs
            max_tardiness = 0  # max tardiness among all finished + just-being-assigned jobs

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
                    if t == 0:
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
                    # stats.episode_rewards[i_episode] != total_tardiness
                    reward = -1*total_tardiness #+ stats.episode_rewards[i_episode]
                    stats.episode_rewards[i_episode] += reward

                    # April 22, 2020-use max_tardiness to represent the result
                    max_tardiness = max_tardiness if tardi < max_tardiness else tardi
                    # April 26: enable the option of min total tardiness
                    #if self.obj == 1:
                    #    stats.episode_obj[i_episode] = max_tardiness
                    #else:
                    stats.episode_obj[i_episode] = total_tardiness

                    # done is True if episode terminated
                    if done:
                        #print("All jobs finished")
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
                    best_next_action = self.choose_action(Q, next_state)# np.argmax(Q[next_state])
                    td_target = reward + self.discount_factor * Q[next_state][best_next_action]
                    td_delta = td_target - Q[self.env.state][action]
                    #for i in range(len(Q)):
                    #    for j in range(len(Q[i])):
                    Q += self.alpha * td_delta * eligibility

                    self.env.state = next_state
                    action = best_next_action

            # EDIT: May 15, 2020, store total jobs into file when one episode finishes
            event_simu.store_arrived_job()
        return Q, stats

if __name__ == '__main__':
    matplotlib.style.use('ggplot')
    plotting = Plotting()
    env = JSPEnv2()
    _conf = ConfigParser()
    res = _conf.read('/Users/yuanyuanli/PycharmProjects/RL-RealtimeScheduling/realtime_jsp'
                     '/etc/app.ini')
    print(res)
    #  Train the model
    sarsa_train = SARSA_L(env, _conf)
    # sarsa_train.criteria = 1

    # plotting.plot_episode_stats(stats)
    filename = '/Users/yuanyuanli/PycharmProjects/RL-RealtimeScheduling/realtime_jsp/results/' \
               'QL_JIT_5000.txt'
               #'QL_lateness_2500.txt'
               #'QLV3_2500.txt'
    # '1000V3.txt'
    time = [5000]#[1000, 5000]
    with open(filename, 'a') as f:
        for t in time:
            sarsa_train.size_time_steps = t
            for c in sarsa_train.criterion:
                sarsa_train.criteria = c
                # event simulator is not fixed
                Q, stats = sarsa_train.learn(plotting)
                # print("stats ", stats)
                # plotting.plot_episode_stats(stats)

                # event simulator is fixed
                # test the model with calculated Q
                # Q_learn.num_episodes = 10
                Q2, stats2 = sarsa_train.fixed_seed(Q, plotting)
                print("New Stats", stats2)
                cri = ""
                if sarsa_train.criteria == 1:
                    cri = "DD"
                elif sarsa_train.criteria == 2:
                    cri = "DD_pt"
                else:
                    cri = "random"
                s = "SarsaL " + str(t) + " " + cri  # +" "+str(lambda_value)
                f.write(s)
                f.write(" ")
                b = np.matrix(stats2.episode_obj)
                np.savetxt(f, b, fmt="%d")
                f.write("\n")
    print("Finished SarsaL")

