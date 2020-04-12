class Machine():

    def __init__(self):
        self.assigned_job = None
        self.idle = True

    def reset(self):
        self.assigned_job = None
        self.idle = True

    def process_job(self, assigned_job, start_processing_t):
        # if self.idle:
        print("Machine is assigned new job ", assigned_job.to_string())
        self.assigned_job = assigned_job
        self.assigned_job.start_processing_t = start_processing_t
        self.idle = False
