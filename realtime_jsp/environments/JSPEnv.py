import gym
from gym import spaces
import numpy as np
import random_self
from realtime_jsp.environments.machines.machine import Machine
from realtime_jsp.simulators.eventsimulator import EventSimulator
from configparser import ConfigParser

class JSPEnv(gym.Env):
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
        self.min_interarrival = 5
        self.min_pt = 30
        self.max_interarrival = 8
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
        self.raw_pt = 0
        self.remain_raw_pt = 0

        # TO MODIFY: use exponential distribution
        self.low = np.array([self.min_num_process, self.min_num_wait, self.min_num_travel], dtype=np.float32)
        self.high = np.array([self.max_num_process, self.max_num_wait, self.max_num_travel], dtype=np.float32)

        # jobs in the related queue make up an action set (now I consider only a machine, so 1-d array is fine)
        self.action_space = [] # spaces.Discrete(2)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)
        self.machine_size = 1
        # self.machines = []

        # April 13, create a map mapping state id to the complete state
        self.state_dict = {}

    '''
    Section 3.1 Decisions are made while 1) a job arrives at an idle machine; 2) a machine with a non-empty queue
    becomes idle. We call these points of time decision epochs. Contrarily, when a job is released or arrives at
    an occupied machine or a machine with an empty queue becomes idle, no decisions are made. We call
    these moments non-decision epochs. The time between two successive epochs is random.
    '''

    # TO MODIFY: current version doesnt consider the change of y in state since there is only a machine
    def step(self, action, events, t):
        # assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        # TO DO: change to multiple machines
        to_release, released_new_jobs, next_interarrival_time, idles = events[0], events[1], events[2], events[3]
        x = self.state[0]
        y = self.state[1]
        z = self.state[2]
        i = self.state[3] # id
        # print("Debug x ", x)
        if to_release:
            for job in released_new_jobs:
                x.append(job)
            print("Debug x after releasing ", x)
        # dummy action: no job is selected to process and the machine keeps idle
        if action == 0:
            dummy = True
            # dummy action, so even machine idle (the update to machine status has been done
            # in event simulator), no selection
            # self.state = self.create_jobs(dummy, self.state)
            r_a = 0

        # non dummy action
        else:
            # TO DO: add transition functions (7 possible states)
            dummy = False
            # TO Modify: if over one machine becomes idle?
            # notice now just consider there is one machine, and all jobs can be processed on the machine
            # print("Debug, the size is ", len(z))
            m = z[0]
            r_a = 0
            # print("Debug now ", x[action], action, "type of x[] ", type(x[action]))
            if m.idle:
                # action maps which job to select
                print("Machine is idle. Debug action ", action, " job is ", x[action].to_string())
                m.process_job(x[action], t)
                z[0] = m
                del x[action]
                release_time = m.assigned_job.release_time
                numerator = self.real_num*self.remain_raw_pt
                denominator = release_time + self.real_num*self.raw_pt - t
                r_a = numerator/denominator

        self.state = np.array([x, y, z, i+1])
        # self.state = self.get_event_state(dummy, self.state)
        # print("Check x ", x, " end ")
        num_travel = len(self.state[0])
        num_wait = len(self.state[1])
        num_process = 0
        for m in z:
            num_process = num_process+1 if m.idle==False else num_process
        sum_num = num_wait + num_process + num_travel
        r_s = -1 * self.price * sum_num
        reward = r_a + r_s

        print("Getting reward r_a ", r_a, " r_s ", r_s, " from action ", action)

        done = bool(num_wait == 0 and num_travel == 0)
        return np.array(self.state), reward, done, {}

    def reset(self, simulator, i):
        # just consider job release for now
        num_wait = 0 # random.randint(1, 6)
        # initialize num_wait new jobs
        wait_jobs = []# self.create_jobs(num_wait)
        machines = []
        # initialize machines and its job being processed
        # reset from time t = 0
        t = 0
        for m in range(self.machine_size):
            machine = Machine()
            job = simulator.arrive_new_job(t, 1)#self.create_jobs(1)
            # assume the job is new, so remain_raw_pt = raw_pt
            self.remain_raw_pt += job[0].pt
            self.raw_pt = self.remain_raw_pt
            machine.process_job(job[0], 0)
            machines.append(machine)
        # initialize jobs processed on machines
        # num_process = random.randint(0, 1)
        # process_jobs = self.create_jobs(num_process)
        num_travel = random_self.randint(0, 6)
        travel_jobs = simulator.arrive_new_job(t, num_travel) # self.create_jobs(num_travel)
        # print("Debug created machine with size ", len(machines))

        self.state = np.array([travel_jobs, wait_jobs, machines, i])  # np.array([num_wait, num_process, num_travel])
        self.state_dict[i] = self.state
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
    env = JSPEnv()
    i = 0
    for i_episode in range(1):
        # iterate every time step to check if there is
        # 1) a job arrives at an idle machine;
        # 2) a machine with a non-empty queue becomes idle
        _conf = ConfigParser()
        _conf.read('app.ini')
        event_simu = EventSimulator(_conf)
        observation = env.reset(event_simu, i)
        granularity = 1 # for calculating the remaining pt
        for t in range(100):
            print("Observation[0] is ", observation[0])
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
            # jobs in the related queue at a decision epoch make up an action set （
            # plus a dummy action 0）
            # if no released and waited job, then dummy action
            if len(env.state[0]) == 0 and len(env.state[1]) == 0:
                action = 0
            else:
                env.action_space = spaces.Discrete(len(env.state[0])) # change to state[1] later
                action = env.action_space.sample() # randomly select
            observation, reward, done, info = env.step(action, events, t)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break
            i = env.state[3] # update id, useful for the reset()
    print("Getting reward ", reward, " state id ", env.state[3])
    print("Finished!")
