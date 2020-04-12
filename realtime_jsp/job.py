class Job():

    def __init__(self, release_time, pt, start_processing_t):
        self.release_time = release_time
        self.pt = pt
        # self.remaining_pt = remaining_pt
        # if current_time - start_processing_pt = pt, it means finishing
        self.start_processing_t = start_processing_t
       # print("Intialize done.")

    def to_string(self):
        job_string = "Job release time is " + str(self.release_time) + " processing time is " + str(self.pt) + " start processing time is "+\
                     str(self.start_processing_t)
        # print("Job release time is {}, processing time is {}, start processing time is {}".
        #      format(self.release_time, self.pt, self.start_processing_t))
        return job_string
