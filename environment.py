import gym
import numpy as np
from frame_preprocessing import FrameProcessor
import random

class AtariEnvironment:
    def __init__(self, env_name, frame_stack_length):
        self.__env = gym.make(env_name)
        self.frame_stack_length = frame_stack_length
        self.action_number = self.__env.action_space.n
        self.fire_action = 1
        self.frame_preprocessor = FrameProcessor()
        self.current_state = None
        self.remaining_lives = 0

    def reset_environment(self, hard_reset = True):
        if hard_reset:
            frame = self.__env.reset()
            self.remaining_lives = 0 #ne znam kako da dobavim broj zivota nakon reset-a, ali svakako ce se u commit_action dobiti pravi broj zivota
                                    # pre nego sto se zivot izgubi
        terminal_life_lost = True
        i = random.randint(1, 10)
        for _ in range(i):
            frame, _, _, _ = self.__env.step(1)  # Action 'Fire'
        processed_frame = self.frame_preprocessor.preprocess(frame)  # (★★★)
        self.current_state = np.repeat(processed_frame, self.frame_stack_length, axis=2) # prvo iskustvo je sacinjeno od 4 ista frejma
        return terminal_life_lost

    def commit_action(self, action):
        next_frame, reward, terminal, info = self.__env.step(action)

        #terminal koji vrati env.step je True samo ako se igra potpuno zavrsila, a ne i ako smo izgubili zivot!
        #kada cuvamo frejm u replay memoriju, bitno je da znamo da li je zivot izgubljen ili ne
        #kako bi mogli da napravimo ispravna stanja za minibatch update
        #naredni if-else omogucava da zabelezimo kada se izgubi zivot
        if info['ale.lives'] < self.remaining_lives:
            terminal_life_lost = True
        else:
            terminal_life_lost = terminal
        self.remaining_lives = info['ale.lives']
        processed_next_frame = self.frame_preprocessor.preprocess(next_frame)
        new_state = np.append(self.current_state[:,:,1:], processed_next_frame, axis=2)
        self.current_state = new_state
        return processed_next_frame, reward, terminal, terminal_life_lost

    def render(self):
        self.__env.render()

    def close(self):
        self.__env.close()

