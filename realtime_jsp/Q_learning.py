import numpy as np
import itertools
import matplotlib
import matplotlib.style
from collections import defaultdict
from event_simulator import Event_simulator
from configparser import ConfigParser
from JSPEnv import JSPEnv
import plotting

'''
Based on: https://www.geeksforgeeks.org/q-learning-in-python/
'''

class q_learning_funcs():

    def __init__(self, env, settings, num_episodes):
        self.env = env
        self.settings = settings
        self.epsilon = float(settings.get('Q_learning', 'epsilon'))
        self.discount_factor = float(settings.get('Q_learning', 'discount_factor'))
        self.alpha = float(settings.get('Q_learning', 'alpha'))
        self.num_episodes = num_episodes

    # Make the $\epsilon$-greedy policy
    def create_epsilon_greedy_policy(self, Q, num_actions):
        """
        Creates an epsilon-greedy policy based
        on a given Q-function and epsilon.

        Returns a function that takes the state
        as an input and returns the probabilities
        for each action in the form of a numpy array
        of length of the action space(set of possible actions).
        """
        def policy_function(state_id):
            action_probabilities = np.ones(num_actions, dtype=float)*self.epsilon/num_actions
            # print("Type ", type(Q[state]))
            best_action = np.argmax(Q[state_id])
            action_probabilities[best_action] += (1.0-self.epsilon)
            return action_probabilities

        return policy_function

    # Build Q-Learning Model
    def q_learning(self):
        """
        Q-Learning algorithm: Off-policy TD control.
        Finds the optimal greedy policy while improving
        following an epsilon-greedy policy"""


        # Keeps track of useful statistics
        stats = plotting.EpisodeStats(
            episode_lengths=np.zeros(self.num_episodes),
            episode_rewards=np.zeros(self.num_episodes))

        event_simu = Event_simulator(self.settings)

        # For every episode
        for i_episode in range(self.num_episodes):
            # Reset the environment and pick the first action
            env.state = self.env.reset(event_simu)
            # Create an epsilon greedy policy function
            # appropriately for environment action space

            granularity = 1
            num_action = len(env.state[0])+1
            # Action value function
            # A nested dictionary that maps
            # state -> (action -> action-value).
            Q = defaultdict(lambda: np.zeros(num_action))
            for t in itertools.count():

                # Q = defaultdict(lambda: np.zeros(self.env.action_space.n))
                # Check decision epoch according to events
                # job release/job arrival (simulation strategy to be used?)
                # /machine idle
                # env.state[2] is machine list
                events = event_simu.event_simulation(t, env.state[2], granularity)
                # update pt
                released_new_jobs = events[1]
                for new_job in released_new_jobs:
                    env.raw_pt += new_job.pt
                machines = events[2]
                env.state[2] = machines
                env.remain_raw_pt -= events[3]

                # get probabilities of all actions from current state
                # if no released and waited job, then dummy action
                if len(env.state[0]) == 0 and len(env.state[1]) == 0:
                    action = 0
                else:
                    policy = self.create_epsilon_greedy_policy(Q, num_action) # plus dummy action
                    action_probabilities = policy(env.state[3])

                    # choose action according to
                    # the probability distribution
                    action = np.random.choice(np.arange(
                        len(action_probabilities)),
                        p=action_probabilities)

                # take action and get reward, transit to next state
                next_state, reward, done, _ = self.env.step(action, events, t)

                # update action size
                num_action = next_state[0]


                # Update statistics
                stats.episode_rewards[i_episode] += reward
                stats.episode_lengths[i_episode] = t

                # TD Update
                best_next_action = np.argmax(Q[next_state[3]])
                td_target = reward + self.discount_factor * Q[next_state[3]][best_next_action]
                print("type is ", type(Q), " action ", action, " Q ", Q, " env.state[3] is ", env.state[3])
                td_delta = td_target - Q[env.state[3]][action]
                Q[env.state[3]][action] += self.alpha * td_delta
                print("Now Q is ", Q)

                # done is True if episode terminated
                if done:
                    break

                env.state = next_state

        return Q, stats


if __name__ == '__main__':
    matplotlib.style.use('ggplot')
    env = JSPEnv()
    _conf = ConfigParser()
    _conf.read('app.ini')
    num_episode = 1#1000
    #  Train the model
    Q_learn = q_learning_funcs(env, _conf, num_episode)
    Q, stats = Q_learn.q_learning()
    plotting.plot_episode_stats(stats)
