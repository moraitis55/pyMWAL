import os

import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")
# env.reset()

LEARNING_RATE = 0.1  # has to be in [0,1] , 0.1 default
DISCOUNT = 0.95  # gamma (how much we value future rewards over present rewards.) -  0.5 default
EPISODES = 25000

SHOW_EVERY = 500

# print(env.observation_space.high) # [0.6 0.07]
# print(env.observation_space.low)  # [-1.2 -0.07]
# print(env.action_space.n)

# size of the Q-table (hardcoded value ~ a real agent won't have this hardcoded because it will be changed by the
# environment)
# we want to be a size at least manageable
DISCRETE_OS_SIZE = [40] * len(env.observation_space.high)
# we have that {-1.2 , 0.6} and {-0.07, 0.07} are the feature ranges of position of the car and velocity. We want to
# separate that range into twenty chunks-buckets-discrete values.
# Now we need to know how big are those chunks
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE
# print(discrete_os_win_size)  # [0.09 0.007]

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2  # double div is python 2 type devision, where it will always divide out to an intenger so we don't have a float
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

'''
Filling the table with values

About the low-high: rewards are either -1 or 0 if we arrive at the flag point.
That's why we use that value range in the table.

About the size: we did define the DISCRETE_OS_SIZE [20 20], so we give that in order to satisfy every possible
combination of position-velocity. Then we need values for every single action that's possible, so the way we are going
to do that is to add another dimension and have an overall size of (20 20 3)
'''
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

ep_rewards = []
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}


# print(q_table.shape)  # (20 20 3)
# print(q_table)

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))


for episode in range(EPISODES):
    episode_reward = 0
    if episode % SHOW_EVERY == 0:
        print(episode)
        render = True
    else:
        render = False
    # example of above get_discrete_state function
    discrete_state = get_discrete_state(env.reset())
    # print(discrete_state)  # [7, 10]
    # print(q_table[discrete_state])  # we can pass discrete_state because it is a tuple, gives [-1.4122215 -0.2093237
    # # -0.77109107]
    # print(np.argmax(q_table[discrete_state]))  # gives -->  0 (this is the maximum action in our state in terms of value)

    done = False

    while not done:
        if np.random.random() > epsilon:  # np.random.random creates a random float between (0,1)
            # action = 2
            # we changed the hardcoded price to this:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)
        new_state, reward, done, _ = env.step(action)
        episode_reward += reward

        # get new discrete state
        new_discrete_state = get_discrete_state(new_state)

        if render:
            # print(new_state, reward)
            env.render()
        if not done:
            max_future_q = np.max(q_table[[new_discrete_state]])  # estimate of optimal future value (see equation)
            current_q = q_table[
                discrete_state + (action,)]  # old value (see equation) - also here I remind that discrete
            # value is a tuple, let's say [20 20], so by adding the action we have a specific value from the table Q
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)  # formula
            q_table[
                discrete_state + (action,)] = new_q  # update the table. IMPORTANT: Take a note that here we update the
            # table in the discrete_state AFTER we took the step (action) of that discrete state -> line where
            # env.step(action)
        # if we are done, either we took 25000 step and we are not getting anywhere, but we might have get the precious
        # position in the environment already
        elif new_state[0] >= env.goal_position:
            print("We made it on episode:" + str(episode))
            q_table[discrete_state + (action,)] = 0  # our reward for completing things

        # and the loop goes on and on..
        discrete_state = new_discrete_state

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
    ep_rewards.append(episode_reward)

    if not episode % SHOW_EVERY:
        # np.save(os.path.join("sendex_chepoint", str(episode) + "-qtable.npy"), q_table)
        average_reward = sum(ep_rewards[-SHOW_EVERY:]) / len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))

        print("Episode:" + str(episode) + "avg:" + str(average_reward) + "min:" + str(
            min(ep_rewards[-SHOW_EVERY:])) + "max:" + str(max(ep_rewards[-SHOW_EVERY:])))
    np.save(os.path.join("sendex_chepoint", str(episode) + "-qtable.npy"), q_table)
env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="avg")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max")
plt.legend(loc=4)
plt.grid(True)
plt.show()
