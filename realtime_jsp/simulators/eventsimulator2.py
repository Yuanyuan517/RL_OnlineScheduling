import numpy as np
from realtime_jsp.environments.jobs.jobdue import JobDue


class EventSimulator2:
    # job release and arrival are predicted from exponential distribution
    # machine idle is calculated from the existing processing status
    def __init__(self, settings):
        self.interarrival_mean_time = int(settings.get('event', 'interarrival_time'))
        self.processing_mean_time = int(settings.get('job', 'processing_time'))
        self.min_processing_time = int(settings.get('job', 'min_processing_time'))
        self.max_processing_time = int(settings.get('job', 'max_processing_time'))
        self.min_num_release_job = int(settings.get('job', 'min_num_release_job'))
        self.max_num_release_job = int(settings.get('job', 'max_num_release_job'))
        self.job_released = False
        self.last_release_time = 0
        self.interarrival_time = 0
        self.current_time = 0
        self.seed = 0
        self.random = True

    def reset(self):
        self.last_release_time = 0
        self.interarrival_time = 0
        self.current_time = 0
        self.seed = 0

    def set_randomness(self, random):
        self.random = random

    def set_seed(self, seed):
        self.seed = seed

    def event_simulation(self, t, machine, granularity):
        self.current_time = t
        to_release = self.check_job_release()
        released_new_jobs = []
        if to_release:
            # number of release job
            if not self.random:
                np.random.seed(self.seed)
            num = np.random.randint(self.min_num_release_job, self.max_num_release_job, 1)
            released_new_jobs = self.release_new_job(num[0])
            next_interarrival_time = self.get_interarrival_time()
            # update time status
            self.interarrival_time = next_interarrival_time
            self.last_release_time = t
        updated_machine, processed_pt, tardiness = self.check_machine_idle_and_update(machine, granularity)
        return np.array([to_release, released_new_jobs, updated_machine, processed_pt, tardiness])

    # release one job at a time, it can be modified later
    def release_new_job(self, number):
        jobs = []

        if not self.random:
            np.random.seed(self.seed)
        # parameter: scale = inverse of the rate parameter = mean
        pts = np.random.randint(self.min_processing_time, self.max_processing_time, number)
            # pt = int(np.random.exponential(self.processing_mean_time, 1))
        if not self.random:
            np.random.seed(self.seed)
        ints = np.random.exponential(self.max_processing_time, number)
        for i in range(number):
            due_t = int(ints[i]) +pts[i]+ self.current_time+1  # Can be modified
            job = JobDue(due_t, pts[i], -1)
            print("Released job ", due_t, " ", pts[i], " at time ", self.current_time)
            jobs.append(job)
        return jobs

    def check_job_release(self):
        if self.last_release_time + self.interarrival_time == self.current_time:
            #print("Release")
            return True
        #print("Not Release")
        return False

    def get_interarrival_time(self):
        if not self.random:
            np.random.seed(self.seed)
        num = int(np.random.exponential(self.interarrival_mean_time, 1))
        print("interrival is ", num)
        return num

    # TO DO: add arrival like release


    # according to the job assigned to the machine, if job is completed, then the machine is idle
    def check_machine_idle(self, machines):
        idles = []
        for m in machines:
            if m.idle:
                idle = True
            else:
                idle = True if self.current_time - m.assigned_job.start_processing_pt == 0 else False
                # idle = True if m.assigned_job.remaining_pt == 0 else False
            idles.append(idle)
        return idles

    def check_machine_idle_and_update(self, machine, granularity):
        processed_pt = 0
        tardiness = 0
        # print("In event_simu, machine num is ", len(machines))
        if machine.idle != True:
            job = machine.assigned_job
            remained_pt = job.pt - (self.current_time - job.start_processing_t)
            if remained_pt <= 0:
                processed_pt += granularity
                tardiness = self.current_time - job.due_t + remained_pt
                machine.reset()

           # print("Debug simulator， remained pt is ", remained_pt, " machine is idle? ", machine.idle)
        updated_machine = machine
        return updated_machine, processed_pt, tardiness

