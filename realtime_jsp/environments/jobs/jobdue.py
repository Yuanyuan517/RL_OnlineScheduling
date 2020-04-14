from .base import Base


class JobDue(Base):

    def __init__(self, due_t, pt, start_processing_t):
        Base.__init__(self, pt, start_processing_t)
        self.due_t = due_t

       # print("Intialize done.")

    def to_string(self):
        job_string = " processing time is " + str(self.pt) + " due time is " + \
                     str(self.due_t)
        # print("Job release time is {}, processing time is {}, start processing time is {}".
        #      format(self.release_time, self.pt, self.start_processing_t))
        return job_string
