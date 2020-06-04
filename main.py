from environment import AtariEnvironment
from exploit_explore import ActionSelector
from network_model import DQN
from replay_memory import ReplayMemory
from target_network_updater import TargetDqnUpdater
from hyperparameters import *
from keras.utils import to_categorical
import numpy as np
import time

def clip(reward):
    if reward > 0:
        return 1
    elif reward < 0:
        return -1
    else:
        return 0

def train():
    environment = AtariEnvironment(env_name=ENV_NAME, frame_stack_length=FRAME_STACK_LENGTH)
    main_dqn = DQN(num_actions=environment.action_number, frame_height=FRAME_HEIGHT, frame_width=FRAME_WIDTH,
                   frame_stack_length=FRAME_STACK_LENGTH, hidden=HIDDEN, batch_size=BATCH_SIZE, path=PATH_READ, path2=PATH_WRITE)
    target_dqn = DQN(num_actions=environment.action_number, frame_height=FRAME_HEIGHT, frame_width=FRAME_WIDTH,
                     frame_stack_length=FRAME_STACK_LENGTH, hidden=HIDDEN, batch_size=BATCH_SIZE, path=PATH_READ, path2=PATH_WRITE)
    replay_memory = ReplayMemory(size=MEMORY_SIZE, frame_height=FRAME_HEIGHT, frame_width=FRAME_WIDTH,
                                 frame_stack_length=FRAME_STACK_LENGTH, batch_size=BATCH_SIZE)
    action_selector = ActionSelector(dqn=main_dqn, num_actions=environment.action_number,
                                     initial_epsilon=EPSILON_INITIAL, middle_epsilon=EPSILON_SECOND,
                                     finish_epsilon=EPSILON_FINAL, minimum_replay_size=REPLAY_MEMORY_START_SIZE,
                                     maximum_replay_size=MEMORY_SIZE, final_frame_number=MAX_FRAMES)
    target_dqn_updater = TargetDqnUpdater(main_dqn=main_dqn, target_dqn=target_dqn)

    total_frame_number = 0
    rewards_per_episode = {}
    frames_per_episode = {}
    episode = 0

    average_last_100_frames = 0
    average_last_100_reward = 0
    best_score = 0
    open('scores/best_scores.txt', 'w').close()
    open('scores/averages.txt', 'w').close()

    #main_dqn.load_model(400)
    #target_dqn.load_model(400)
    while total_frame_number < MAX_FRAMES:
        episode += 1

        rewards_per_episode[episode] = 0
        frames_per_episode[episode] = 0

        terminal_life_lost = environment.reset_environment(hard_reset=True)

        while frames_per_episode[episode] < MAX_EPISODE_LENGTH:
            action = 1 if terminal_life_lost else action_selector.act(environment.current_state, total_frame_number)
            processed_next_frame, reward, terminal, terminal_life_lost = environment.commit_action(action)
            replay_memory.add_experience(action, processed_next_frame[:,:,0], clip(reward), terminal_life_lost)
            if REPLAY_MEMORY_START_SIZE < total_frame_number:
                if total_frame_number % UPDATE_FREQ == 0:
                    states, actions, rewards, next_states, terminals = replay_memory.sample_minibatch()

                    # calculate best actions in next states based on main dqn!
                    best_actions = main_dqn.get_best_actions_batch(next_states)

                    #calculate q_values of these actions in next states based on target network!
                    #firstly, one hot encode best found actions
                    ohe_best_actions_next_states = to_categorical(best_actions, num_classes=environment.action_number)
                    ohe_best_actions_current_states = to_categorical(actions, num_classes=environment.action_number)
                    next_states_q_values = target_dqn.predict_batch(next_states, ohe_best_actions_next_states)
                    next_states_best_q_value = np.sum(next_states_q_values, axis=1)

                    #the Bellman update ->
                    target_q_values = rewards + (1 - terminals) * DISCOUNT_FACTOR * next_states_best_q_value

                    #gradient descent
                    main_dqn.fit_batch(states, ohe_best_actions_current_states,
                                       ohe_best_actions_current_states * np.expand_dims(target_q_values, axis=1))

                if total_frame_number % NETW_UPDATE_FREQ == 0:
                    target_dqn_updater.update_target_network()

            total_frame_number += 1
            frames_per_episode[episode] += 1
            rewards_per_episode[episode] += reward
            if terminal:
                break
            if terminal_life_lost:
                terminal_life_lost = environment.reset_environment(hard_reset=False) #NAKON SVAKOG IZGUBLJENOG ZIVOTA PUCAJ SLUCAJNO KAKO BI SE STVORIO U NOVOJ SITUACIJI

        print("\nEpisode %d ended." % episode)
        print("Reward: %d" % rewards_per_episode[episode])
        print("Frames: %d" % frames_per_episode[episode])
        print("Replay memory size: %d" % replay_memory.get_size())
        print("Current epsilon: %5f\n" % action_selector.eps_debug)

        average_last_100_reward += rewards_per_episode[episode]
        average_last_100_frames += frames_per_episode[episode]

        if best_score < rewards_per_episode[episode]:
            best_score = rewards_per_episode[episode]
            file = open("scores/best_scores.txt", 'a')
            file.write("Episode: " + str(episode) + " | New best score: " + str(best_score) + "\n")
            file.close()

        if episode % 100 == 0:
            file = open("scores/averages.txt", 'a')
            file.write("\nEpisodes %d - %d results:" % (episode - 100, episode))
            average_last_100_reward /= 100
            average_last_100_frames /= 100
            file.write("\nAverage reward per episode: %.5f" % average_last_100_reward)
            file.write("\nAverage frames per episode: %.2f\n" % average_last_100_frames)
            file.close()

            average_last_100_reward = 0
            average_last_100_frames = 0

            if total_frame_number > REPLAY_MEMORY_START_SIZE:
                main_dqn.save_model(episode)

def test():
    environment = AtariEnvironment(env_name=ENV_NAME, frame_stack_length=FRAME_STACK_LENGTH)
    main_dqn = DQN(num_actions=environment.action_number, frame_height=FRAME_HEIGHT, frame_width=FRAME_WIDTH,
                   frame_stack_length=FRAME_STACK_LENGTH, hidden=HIDDEN, batch_size=BATCH_SIZE, path=PATH_READ,
                   path2=PATH_WRITE)
    action_selector = ActionSelector(dqn=main_dqn, num_actions=environment.action_number,
                                     initial_epsilon=EPSILON_INITIAL, middle_epsilon=EPSILON_SECOND,
                                     finish_epsilon=EPSILON_FINAL, minimum_replay_size=REPLAY_MEMORY_START_SIZE,
                                     maximum_replay_size=MEMORY_SIZE)

    main_dqn.load_model(7800)
    print(environment.action_number)
    terminal_life_lost = environment.reset_environment(hard_reset=True)
    game_reward = 0
    while True:
        environment.render()
        #time.sleep(.03)
        action = 1 if terminal_life_lost else action_selector.act_test(environment.current_state)
        processed_next_frame, reward, terminal, terminal_life_lost = environment.commit_action(action)
        game_reward += reward
        if terminal_life_lost:
            environment.reset_environment(hard_reset=False)
        if terminal:
            print("Game reward: " + str(game_reward))
            game_reward = 0
            terminal_life_lost = environment.reset_environment(hard_reset=True)


if __name__ == '__main__':
    #train()
    test()
    """import gym
    from gym.utils.play import play

    env = gym.make("BreakoutDeterministic-v4")
    play(env, zoom=4)"""