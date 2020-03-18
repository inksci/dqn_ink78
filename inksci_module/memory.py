import numpy as np

class MEMORY():
    def __init__(self, MEMORY_CAPACITY, s_dim, a_dim):
        self.__version__ = "2.1"
        self.pointer = 0 # increase without limitation
        self.MEMORY_CAPACITY = MEMORY_CAPACITY

        self.pre_states = np.zeros((MEMORY_CAPACITY, s_dim), dtype=np.float32)
        self.actions = np.zeros((MEMORY_CAPACITY, a_dim), dtype=np.float32)
        self.post_states = np.zeros((MEMORY_CAPACITY, s_dim), dtype=np.float32)
        self.rewards = np.zeros((MEMORY_CAPACITY, 1), dtype=np.float32)

        # version: 2.1
        self.dones = np.zeros((MEMORY_CAPACITY, 1), dtype=np.int)
        
    def add(self, prestate, action, reward, poststate, done):
        
        index = self.pointer % self.MEMORY_CAPACITY

        self.pre_states[index, :] = prestate
        self.actions[index, :] = action
        self.rewards[index, :] = reward
        self.post_states[index, :] = poststate

        # version: 2.1
        self.dones[index, :] = done
        
        self.pointer += 1
        
    def sample(self, batch_size):
        indexes = np.random.choice( min(self.pointer, self.MEMORY_CAPACITY), size=batch_size)
        
        pre_states = self.pre_states[indexes]
        actions = self.actions[indexes]
        rewards = self.rewards[indexes]
        post_states = self.post_states[indexes]
        dones = self.dones[indexes]
        
        return pre_states, actions, rewards, post_states, dones
        
    # version: 2.0
    def save(self, data_save_dir):
        import os
        if not os.path.exists(data_save_dir):
            os.mkdir(data_save_dir)

        np.save(data_save_dir+"/memory.pointer", [self.pointer])
        np.save(data_save_dir+"/memory.pre_states", self.pre_states)
        np.save(data_save_dir+"/memory.actions", self.actions)
        np.save(data_save_dir+"/memory.post_states", self.post_states)
        np.save(data_save_dir+"/memory.rewards", self.rewards)
        np.save(data_save_dir+"/memory.dones", self.dones)

    def load(self, data_save_dir):
        self.pointer = np.load(data_save_dir+"/memory.pointer.npy")[0]
        self.pre_states = np.load(data_save_dir+"/memory.pre_states.npy")
        self.actions = np.load(data_save_dir+"/memory.actions.npy")
        self.post_states = np.load(data_save_dir+"/memory.post_states.npy")
        self.rewards = np.load(data_save_dir+"/memory.rewards.npy")
        self.dones = np.load(data_save_dir+"/memory.dones.npy")
