from .base import Base


class JobDue(Base):

    id_counter = 0 # it seems static, so i should create an internal id

    def __init__(self, due_t, pt, start_processing_t):
        Base.__init__(self, pt, start_processing_t)
        self.due_t = due_t
        self.counter = 0

    def setID(self):
        JobDue.id_counter += 1
        self.counter = JobDue.id_counter
        # print("Set counter", self.counter)

       # print("Intialize done.")

    def to_string(self):
        job_string = "Id "+ str(self.counter) +" processing time is " + str(self.pt) + " due time is " + \
                     str(self.due_t)
        # print("Job release time is {}, processing time is {}, start processing time is {}".
        #      format(self.release_time, self.pt, self.start_processing_t))
        return job_string
