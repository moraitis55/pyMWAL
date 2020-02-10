from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from collections import deque
from PIL import Image
from tqdm import tqdm
import time
import random
import cv2
import os
import numpy as np
import tensorflow as tf

#LOAD_MODEL = "models/256x2pass3____25.000max____3.50avg___-200.00min__1560471815.model"  # or filepath None
LOAD_MODEL = None

# DQN settings
LEARNING_RATE = 0.001  # start 0.001
DENSE_UNITS = 256
HIDDEN_LAYERS = 1
DROPOUT = 0
ACTIVATION_FUNCTION = 'relu'
LOSS_FUNCTION = 'mse'  # tf.losses.humber_loss
DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # how many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # terminal states (end of episodes)
MODEL_NAME = "256x2"
MIN_REWARD = -200  # for model save / player didn't get to the food, but still didn't hit an enemy which is -300 reward
MEMORY_FRACTION = 0.20  # 0.9

# Environment settings
EPISODES = 25000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

# Stats settings
AGGREGATE_STATS_EVERY = 50  # every episodes to aggregate
RENDER_EVERY = 50
SHOW_PREVIEW = False  # see visuals of everything running


class Blob:
    def __init__(self, size):
        self.size = size
        self.x = np.random.randint(0, size)
        self.y = np.random.randint(0, size)

    def __str__(self):
        return "Blob ({0}, {1})".format(self.x, self.y)

    def __sub__(self, other):
        return self.x - other.x, self.y - other.y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def action(self, choice):
        """
        Gives us 9 total movement options. (0,1,2,3,4,5,6,7,8)
        """
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)
        elif choice == 4:
            self.move(x=1, y=0)
        elif choice == 5:
            self.move(x=-1, y=0)
        elif choice == 6:
            self.move(x=0, y=1)
        elif choice == 7:
            self.move(x=0, y=-1)
        elif choice == 8:
            self.move(x=0, y=0)

    def move(self, x=False, y=False):

        # If no value for x, move randomly
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        # If no value for y, move randomly
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > self.size - 1:
            self.x = self.size - 1
        if self.y < 0:
            self.y = 0
        elif self.y > self.size - 1:
            self.y = self.size - 1


class BlobEnv:
    SIZE = 10
    RETURN_IMAGES = True
    MOVE_PENALTY = 1
    ENEMY_PENALTY = 300
    FOOD_REWARD = 25
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)  # 4
    ACTION_SPACE_SIZE = 9
    PLAYER_N = 1  # player key in dict
    FOOD_N = 2  # food key in dict
    ENEMY_N = 3  # enemy key in dict
    # the dict! (colors)
    d = {1: (255, 175, 0),
         2: (0, 255, 0),
         3: (0, 0, 255)}

    def reset(self):
        self.player = Blob(self.SIZE)
        self.food = Blob(self.SIZE)
        while self.food == self.player:
            self.food = Blob(self.SIZE)
        self.enemy = Blob(self.SIZE)
        while self.enemy == self.player or self.enemy == self.food:
            self.enemy = Blob(self.SIZE)

        self.episode_step = 0

        if self.RETURN_IMAGES:
            observation = np.array(self.get_image())
        else:
            observation = (self.player - self.food) + (self.player - self.enemy)
        return observation

    def step(self, action):
        self.episode_step += 1
        self.player.action(action)

        #### MAYBE ###
        # self.enemy.move()
        # self.food.move()
        ##############

        if self.RETURN_IMAGES:
            new_observation = np.array(self.get_image())
        else:
            new_observation = (self.player - self.food) + (self.player - self.enemy)

        if self.player == self.enemy:
            reward = -self.ENEMY_PENALTY
        elif self.player == self.food:
            reward = self.FOOD_REWARD
        else:
            reward = -self.MOVE_PENALTY

        done = False
        if reward == self.FOOD_REWARD or reward == -self.ENEMY_PENALTY or self.episode_step >= 200:
            done = True

        return new_observation, reward, done

    def render(self):
        img = self.get_image()
        img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
        cv2.imshow("image", np.array(img))  # show it!
        cv2.waitKey(1)

    # FOR CNN #
    def get_image(self):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
        env[self.food.x][self.food.y] = self.d[self.FOOD_N]  # sets the food location tile to green color
        env[self.enemy.x][self.enemy.y] = self.d[self.ENEMY_N]  # sets the enemy location to red
        env[self.player.x][self.player.y] = self.d[self.PLAYER_N]  # sets the player tile to blue
        img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
        return img


env = BlobEnv()

# For stats
ep_rewards = [-200]

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# Memory fraction, used mostly when trai8ning multiple agents
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
# backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')

'''
    Note:
    
    Normally Tensorboard wants to create a log file every time we are doing a fit to the model.
    So since we are doing a fit constantly we don't want to create all those log files. So we use that modified version
    of it to create only one log file.
'''


# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        #self.writer = tf.summary.FileWriter(self.log_dir)
        self.writer = tf.summary.create_file_writer(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method to write logs (tensorflow 2.0)
    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step +=1
                self.writer.flush()

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


class DQNAgent:
    def __init__(self):

        # Main model # gets trained every step
        # this model will contain a lot of randomness so we want to have another, more stable, model for predictions
        self.model = self.create_model()

        # Target model # this is what we .predict against every step
        # Notice: nn models are always initialize randomly, so we want to start both models we the exact same weights.
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        '''
            Notes:
            We use to train nn with batches to speed up training but also get better results.
            We don't want to train the network in only one sample (step) because it will adjust its weights only to that
            one think and its like over-fitting to this thing.
            So we use that replay_memory to collect a number of steps and the fit in model to all of these changes at 
            once. 
            
            So we define a replay-memory of 50.000 steps and then take a RANDOM SAMPLING of those 50.000 steps and
            that's the our batch to fit the nn.
        '''
        # used to train model with batches
        # think of it as a list of a max size
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # use this to decide when to update the target model
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{0}-{1}".format(MODEL_NAME, int(time.time())))

        self.target_update_counter = 0

    # Standard dense model
    # Takes observation  space shape as input shape, and action space shape as output shape
    def create_model(self):

        if LOAD_MODEL is not None:
            print("Loading {0}".format(LOAD_MODEL))
            model = load_model(LOAD_MODEL)
            print("Model {0} loaded!".format(LOAD_MODEL))
        else:
            model = Sequential()
            model.add(Conv2D(256, (3, 3), input_shape=env.OBSERVATION_SPACE_VALUES))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(2, 2))
            model.add(Dropout(0.2))

            model.add(Conv2D(256, (3, 3), input_shape=env.OBSERVATION_SPACE_VALUES))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(2, 2))
            model.add(Dropout(0.2))

            model.add(Flatten())
            model.add((Dense(64)))

            model.add(Dense(env.ACTION_SPACE_SIZE, activation="linear"))
            model.compile(loss=LOSS_FUNCTION, optimizer=Adam(lr=LEARNING_RATE), metrics=['accuracy'])
        return model

    #   transition is our observation space, action, reward ->  new observation space (and whether or not it is done)
    #   we need to do that in order to the the new_q formula
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # *state.shape = unpacking state
    # div by 255 to normalize the values (rgb data)
    # model predict always returns a list even though in this case we only predicting against one element it will still
    # return a list of one element, so we still want the zero element of that list
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0]

    """
    Notes:
    Here the first thing we want to do is to check if we actually wanna train. 
    We have the replay_memory which is quite big memory and from this we want to take a mini batch that is relevant small
    but also a decent size in order to avoid over-fitting the model.
    Typically nn want to train in batches of 32 or 64, so we have to do the same here. We want 32 or 64 batch size to be
    pretty small compared to the size of our physical memory. (sentex uses 10000<memory<50000)
    """

    def train(self, terminal_state, step):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch]) / 255  # transition[0]=current_state
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch]) / 255  # trans[3] = new_current_state
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        Y = []

        # Now we need to enumerate our batches
        # calculate the learned value in formula.
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q  # this is the 'learned q value' from formula.
            else:
                new_q = reward  # if we are done, there is no future Q.

            # Update Q value for given state and action taken.
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            Y.append(current_qs)

            # Fit on all samples as one batch, log only on terminal state
            self.model.fit(np.array(X) / 255, np.array(Y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False,
                           callbacks=[self.tensorboard] if terminal_state else None)

            if terminal_state:
                self.target_update_counter += 1

            if self.target_update_counter > UPDATE_TARGET_EVERY:
                self.target_model.set_weights(self.model.get_weights())
                self.target_update_counter = 0


agent = DQNAgent()

# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), unit="episode"):
    # update tensorboard step every episode
    agent.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state = env.reset()

    # Reset flag and start iterating until episode ends
    done = False
    while not done:
        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(agent.get_qs(current_state))
        else:
            # Get random action
            action = np.random.randint(0, env.ACTION_SPACE_SIZE)

        new_state, reward, done = env.step(action)

        # Transform new continuous state to new discrete state and count reward
        episode_reward += reward

        if SHOW_PREVIEW and not episode % RENDER_EVERY:
            env.render()

        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1

        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
                                           epsilon=epsilon)

            # Save model, but only when min reward is greater or equal a set value
            if min_reward >= MIN_REWARD:
                agent.model.save(
                    'models/pass1/{0}____{1}max____{2}avg____{3}min__{4}.model'.format(MODEL_NAME, max_reward, average_reward,
                                                                         min_reward, int(time.time())))

                # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)






