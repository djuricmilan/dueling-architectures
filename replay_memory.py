import numpy as np
import random
import pickle

class ReplayMemory:
    def __init__(self, size=1000000, frame_height=84, frame_width=84, frame_stack_length=4, batch_size = 32):
        self.size = size
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.frame_stack_length = frame_stack_length
        self.batch_size = batch_size

        self.__count = 0
        self.__current = 0

        self.__allocate_memory()

    def __allocate_memory(self):
        self.actions = np.empty(self.size, dtype=np.uint8)
        self.rewards = np.empty(self.size, dtype=np.uint8)
        self.frames = np.empty((self.size, self.frame_height, self.frame_width), dtype=np.uint8)
        self.terminal_flags = np.empty(self.size, dtype=np.uint8)

        self.states = np.empty((self.batch_size, self.frame_stack_length, self.frame_height, self.frame_width),
                               dtype=np.uint8)
        self.new_states = np.empty((self.batch_size, self.frame_stack_length, self.frame_height, self.frame_width),
                               dtype=np.uint8)
        self.indices = np.empty(self.batch_size, dtype=np.int32)

    def add_experience(self, action, frame, reward, terminal):
        self.actions[self.__current] = action
        self.rewards[self.__current] = reward
        self.terminal_flags[self.__current] = terminal
        self.frames[self.__current, ...] = frame

        self.__current += 1
        self.__current %= self.size
        if self.__count < self.size:
            self.__count += 1

    def fill_indices(self):
        #frejmovi se cuvaju linearno, nije svakaa sekvenca od 4 frejma ispravna, jer mozda poticu iz razlicitih partija
        for i in range(self.batch_size):
            while True:
                index = random.randint(self.frame_stack_length, self.__count - 1)
                if index < self.frame_stack_length:
                    #iskustva nisu povezana, preskoci
                    continue
                if index >= self.__current and index - self.frame_stack_length <= self.__current:
                    #iskustva nisu povezana, preskoci
                    continue
                if self.terminal_flags[index - self.frame_stack_length:index].any():
                    # iskustva nisu povezana, preskoci
                    continue
                break
            #za dati indeks se moze kreirati ispravna sekvenca frejmiova
            self.indices[i] = index

    def prepare_state(self, index):
        self.states[index, ...] = self.frames[self.indices[index] - self.frame_stack_length : self.indices[index], ...]

    def prepare_new_state(self, index):
        self.new_states[index, ...] = self.frames[self.indices[index] - self.frame_stack_length + 1 : self.indices[index] + 1, ...]

    def sample_minibatch(self):
        self.fill_indices()
        for i, index in enumerate(self.indices):
            self.prepare_state(i)
            self.prepare_new_state(i)

        # stanja i nova stanja se iz oblika (batchpy_size, frame_stack_length, frame_height, frame_width) menjaju u
        # oblik (batch_size, frame_height, frame_width, frame_stack_length)
        return np.transpose(self.states, axes=(0, 2, 3, 1)), self.actions[self.indices], self.rewards[
            self.indices], np.transpose(self.new_states, axes=(0, 2, 3, 1)), self.terminal_flags[self.indices]

    def get_size(self):
        return self.__count