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
        def policy_function(state):
            action_probabilities = np.ones(self.num_actions, dtype=float)*self.epsilon/num_actions
            best_action = np.argmax(Q[state])
            action_probabilities[best_action] += (1.0-self.epsilon)
            return action_probabilities

        return policy_function

    # Build Q-Learning Model
    def q_learning(self, env):
        """
        Q-Learning algorithm: Off-policy TD control.
        Finds the optimal greedy policy while improving
        following an epsilon-greedy policy"""
        # Action value function
        # A nested dictionary that maps
        # state -> (action -> action-value).
        Q = defaultdict(lambda: np.zeros(self.env.action_space.n))

        # Keeps track of useful statistics
        stats = plotting.EpisodeStats(
            episode_lengths=np.zeros(self.num_episodes),
            episode_rewards=np.zeros(self.num_episodes))

        # Create an epsilon greedy policy function
        # appropriately for environment action space
        policy = self.create_epsilon_greedy_policy(Q, )
        _conf = ConfigParser()
        _conf.read('app.ini')
        event_simu = Event_simulator(_conf)

        # For every episode
        for i_episode in range(self.num_episodes):
            # Reset the environment and pick the first action
            state = self.env.reset(event_simu)
            granularity = 1
            for t in itertools.count():
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
                action_probabilities = policy(state)

                # choose action according to
                # the probability distribution
                action = np.random.choice(np.arange(
                    len(action_probabilities)),
                    p=action_probabilities)

                # take action and get reward, transit to next state
                next_state, reward, done, _ = self.env.step(action, events, t)

                # Update statistics
                stats.episode_rewards[i_episode] += reward
                stats.episode_lengths[i_episode] = t

                # TD Update
                best_next_action = np.argmax(Q[next_state])
                td_target = reward + discount_factor * Q[next_state][best_next_action]
                td_delta = td_target - Q[state][action]
                Q[state][action] += alpha * td_delta

                # done is True if episode terminated
                if done:
                    break

                state = next_state

        return Q, stats


if __name__ == '__main__':
    matplotlib.style.use('ggplot')
    env = JSPEnv()
    #  Train the model
    Q_learn = q_learning_funcs(1000)
    Q_learn.q_learning(env)
