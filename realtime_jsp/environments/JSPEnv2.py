import gym
from gym import spaces
import numpy as np
from realtime_jsp.environments.machines.machine import Machine
from realtime_jsp.simulators.eventsimulator2 import EventSimulator2
from configparser import ConfigParser


class JSPEnv2(gym.Env):
    """
    Description:

        A system contains 1 (consider 5 later) machines and produces 2 products (Pa and Pb) with 2 operation flows. The release
        interarrival times of Pa and Pb are exponentially distributed with mean 5 and 8 time units. The travel
        times and processing times also follow exponential distributions. The objective is to minimize the cycle
        time.

    Personal Thought: When there is only a machine, job release time should = job arrival time

    Source:
        https://www.informs-sim.org/wsc17papers/includes/files/329.pdf

    Observation:
        Type:
        Num	Observation                                                     Min         Max
        0   release interarrival time of products   exp distributed between 5   and     8
        1    pt                                                             30          60

        Type: Box(3)
        Num	Observation                                         Min         Max
        0   (m1)num of jobs being travelling to (x_m)           0           5
        1   (m1)num of jobs being waiting in front of (y_m)     0           5
        2   (m1)num of jobs being processed on (z_m)            0           1
        4   raw processing time
        # * 5 (consider only a machine first)

        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24 deg        24 deg
        3	Pole Velocity At Tip      -Inf            Inf

    Actions:
        Type: Discrete(2)
        Num	Action
        0   select one job to process
        1   no job is selected and the machine keeps idle

    Reward:
        r(s'|s, a) = (1-k)*r_a + k*\deta*r_s, 0<k<1 indicates the relative importance of the parts.
        1st part is the reward for the action selection;
        2nd part is the reward for the state sojourning in a time period \deta (the cost of hplding one job per time unit)
        \deta is the time between 2 successive decision epochs.
        For dummy action, r_a = 0

    Starting State:
        All observations are assigned a random value from the distribution

    Episode Termination:
        TO DO: define a terminate condition

        Pole Angle is more than 12 degrees
        Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    """

    metadata = {}

    def __init__(self):
        self.min_pt = 30
        self.max_pt = 60
        self.min_num_wait = 0
        self.min_num_process = 0
        self.min_num_travel = 0
        self.max_num_wait = 5
        self.max_num_process = 1
        self.max_num_travel = 5
        # termination criteria
        self.time_horizon = 100

        # reward related calculation (to modify)
        self.k = 0.5  # the relative importance of 1st part of reward function
        # the 2nd part is 1-k
        self.price = 0.025  # price of holding one job per time unit
        self.real_num = 2.5

        # TO MODIFY: use exponential distribution
        self.low = np.array([self.min_num_wait], dtype=np.float32)
        self.high = np.array([self.max_num_wait], dtype=np.float32)

        # jobs in the related queue make up an action set (now I consider only a machine, so 1-d array is fine)
        self.action_space = 0 # spaces.Discrete(20) # There wont be over 20 waiting jobs
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.waiting_jobs = []

        # self.machine_size = 1
        self.machine = None

    '''
    Section 3.1 Decisions are made while 1) a job arrives at an idle machine; 2) a machine with a non-empty queue
    becomes idle. We call these points of time decision epochs. Contrarily, when a job is released or arrives at
    an occupied machine or a machine with an empty queue becomes idle, no decisions are made. We call
    these moments non-decision epochs. The time between two successive epochs is random.
    '''

    # no dummy action considered
    # TO MODIFY: current version doesnt consider the change of y in state since there is only a machine
    def step(self, action, events, t):
        # assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        # TO DO: change to multiple machines
        to_release, released_new_jobs, next_interarrival_time, idles = events[0], events[1], events[2], events[3]
        # print("Debug x ", x)
        if to_release:
            for job in released_new_jobs:
                self.waiting_jobs.append(job)
            print("Debug x after releasing ", self.state)
       # if self.machine.idle:
        # action maps which job to select
        print("Machine is idle. Debug action ", action, " waiting size ", len(self.waiting_jobs))
        print("Job is ", self.waiting_jobs[action].to_string())
        self.machine.process_job(self.waiting_jobs[action], t)
        del self.waiting_jobs[action]
        new_state = len(self.waiting_jobs)
        # sort jobs according to the due date, 1st one is the one with smallest due date (urgent)
        self.waiting_jobs.sort(key=self.takeDueTime)

        # calculate current total tardiness as reward
        reward = 0
        for j in self.waiting_jobs:
            tardiness = (t+j.pt)-j.due_t
            if tardiness < 0:
                tardiness = 0
            reward += tardiness
            print("At time ", t, " pt ", j.pt, " due_t ", j.due_t, " tard ", tardiness)
        processed_pt = t - self.machine.assigned_job.start_processing_t
        remained_pt = self.machine.assigned_job.pt - processed_pt
        machine_job_tard = (t+remained_pt) - self.machine.assigned_job.due_t
        if machine_job_tard < 0:
            machine_job_tard = 0
        reward += machine_job_tard
        reward = -1*reward # cz this is tardiness

        print("Getting reward ", reward, " from action ", action)

        done = bool(new_state == 0)
        return new_state, reward, done, {}

    def takeDueTime(self, job):
        return job.due_t+job.pt


    def reset(self, simulator):
        # initialize machines and its job being processed
        # reset from time t = 0
        t = 0
        machine = Machine()
        job = simulator.release_new_job(1)#self.create_jobs(1)
        machine.process_job(job[0], 0)
        self.machine = machine
        # always start from 3 waiting jobs
        self.waiting_jobs = simulator.release_new_job(3)
        # print("Debug created machine with size ", len(machines))
        self.action_space = spaces.Discrete(len(self.waiting_jobs))
        self.state = len(self.waiting_jobs)+1 # plus dummy one  #np.array([travel_jobs, wait_jobs, machines, i])  # np.array([num_wait, num_process, num_travel])

        return self.state

    '''
        def create_jobs(self, num):
        jobs = []
        for i in range(num):
            pt = random.randint(self.min_pt, self.max_pt)
            job = Job(0, pt, pt)
            jobs.append(job)
        return jobs
    '''


if __name__ == '__main__':
    env = JSPEnv2()
    i = 0
    for i_episode in range(1):
        # iterate every time step to check if there is
        # 1) a job arrives at an idle machine;
        # 2) a machine with a non-empty queue becomes idle
        _conf = ConfigParser()
        _conf.read('app.ini')
        event_simu = EventSimulator2(_conf)
        observation = env.reset(event_simu)
        granularity = 1  # for calculating the remaining pt
        for t in range(100):
            print("Observation[0] is ", observation[0])
            # Check decision epoch according to events
            # job release/job arrival (simulation strategy to be used?)
            # /machine idle
            # env.state[2] is machine list
            events = event_simu.event_simulation(t, env.machines, granularity)
            # update pt
            released_new_jobs = events[1]
            for new_job in released_new_jobs:
                env.raw_pt += new_job.pt
            machines = events[2]
            env.state[2] = machines
            env.remain_raw_pt -= events[3]
            # jobs in the related queue at a decision epoch make up an action set （
            # plus a dummy action 0）
            # if no released and waited job, then dummy action
            if len(env.state[0]) == 0 and len(env.state[1]) == 0:
                action = 0
            else:
                env.action_space = spaces.Discrete(len(env.state[0])) # change to state[1] later
                action = env.action_space.sample()  # randomly select
            observation, reward, done, info = env.step(action, events, t)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
            i = env.state[3] # update id, useful for the reset()
    print("Getting reward ", reward, " state id ", env.state[3])
    print("Finished!")
