# Control parameters
ENV_NAME = "BreakoutDeterministic-v4"
MAX_EPISODE_LENGTH = 18000       # Equivalent of 5 minutes of gameplay at 60 frames per second
NETW_UPDATE_FREQ = 1000          # Number of chosen actions between updating the target network.
                                 # According to Mnih et al. 2015 this is measured in the number of
                                 # parameter updates (every four actions), however, in the
                                 # DeepMind code, it is clearly measured in the number
                                 # of actions the agent choses
DISCOUNT_FACTOR = 0.99           # gamma in the Bellman equation
REPLAY_MEMORY_START_SIZE = 50000 # Number of completely random actions,
                                 # before the agent starts learning
MAX_FRAMES = 5000000             # Total number of frames the agent sees
MEMORY_SIZE = 1000000            # Number of transitions stored in the replay memory
FRAME_STACK_LENGTH = 4           # length of frame stack that represents a single state
UPDATE_FREQ = 4                  # Every four actions a gradient descend step is performed
HIDDEN = 1024                    # Number of filters in the final convolutional layer. The output
                                 # has the shape (1,1,1024) which is split into two streams. Both
                                 # the advantage stream and value stream have the shape
                                 # (1,1,512). This is slightly different from the original
                                 # implementation but tests I did with the environment Pong
                                 # have shown that this way the score increases more quickly
BATCH_SIZE = 32                  # Batch size
EPSILON_INITIAL = 1              # initial epsilon exploration factor
EPSILON_SECOND = 0.1             # epsilon exploration factor when replay memory reaches its MEMORY_SIZE
EPSILON_FINAL = 0.01             # final(minimum) epsilon value
FRAME_WIDTH = 84
FRAME_HEIGHT = 84

#PATH = "output"
PATH_READ = "models2"
PATH_WRITE = "models2"