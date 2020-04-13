import numpy as np
from job import Job

class Event_simulator():
    # job release and arrival are predicted from exponential distribution
    # machine idle is calculated from the existing processing status
    def __init__(self, settings):
        self.interarrival_mean_time = int(settings.get('event', 'interarrival_time'))
        self.processing_mean_time = int(settings.get('job', 'processing_time'))
        self.job_released = False
        self.last_release_time = 0
        self.interarrival_time = 0
        np.random.seed(46)

    def event_simulation(self, t, machines, granularity):
        to_release = self.check_job_release(t)
        released_new_jobs = []
        if to_release:
            released_new_jobs = self.release_new_job(t, 1)
            next_interarrival_time = self.get_interarrival_time()
            # update time status
            self.interarrival_time = next_interarrival_time
            self.last_release_time = t
        updated_machines, processed_pt = self.check_machine_idle_and_update(machines, t, granularity)
        return np.array([to_release, released_new_jobs, updated_machines, processed_pt])

    # release one job at a time, it can be modified later
    def release_new_job(self, t, number):
        jobs = []
        for i in range(number):
            # parameter: scale = inverse of the rate parameter = mean
            pt = int(np.random.exponential(self.processing_mean_time, 1))
            job = Job(t, pt, -1)
            jobs.append(job)
        return jobs

    def check_job_release(self, t):
        if self.last_release_time + self.interarrival_time == t:
            return True
        return False

    def get_interarrival_time(self):
        num = int(np.random.exponential(self.interarrival_mean_time, 1))
        print("interrival is ", num)
        return num

    # TO DO: add arrival like release


    # according to the job assigned to the machine, if job is completed, then the machine is idle
    def check_machine_idle(self, machines, t):
        idles = []
        for m in machines:
            if m.idle:
                idle = True
            else:
                idle = True if t - m.assigned_job.start_processing_pt == 0 else False
                # idle = True if m.assigned_job.remaining_pt == 0 else False
            idles.append(idle)
        return idles

    def check_machine_idle_and_update(self, machines, t, granularity):
        updated_machines = []
        processed_pt = 0
        # print("In event_simu, machine num is ", len(machines))
        for m in machines:
            if m.idle != True:
                job = m.assigned_job
                remained_pt = job.pt - (t - job.start_processing_t)
                if remained_pt <= 0:
                    processed_pt += granularity
                    m.reset()
                print("Debug simulator， remained pt is ", remained_pt, " machine is idle? ", m.idle)
            updated_machines.append(m)
        return updated_machines, processed_pt

