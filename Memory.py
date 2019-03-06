import random

class Memory():

    def __init__ (self, max_memory):
        self.max_memory = max_memory
        self.mem = []

    def add_sample (self, state, action, reward, next_state, done):
        sample = [state, action, reward, next_state, done]
        
        if (len(self.mem) <= self.max_memory):
            self.mem.append(sample)
        else:
            self.mem.pop(0)
            self.mem.append(sample)

    def get_batch (self, number):
        return random.sample(self.mem, min(len(self.mem), number))