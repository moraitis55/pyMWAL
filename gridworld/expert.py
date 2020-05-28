import ast
import csv
import datetime

from tqdm import tqdm
import numpy as np
from gridworld.grid import GridState
from transition import ThetaEstimator


def trajectory_generator(model):
    with open('gridworld/expert_trajectories/' + model + '.csv', 'r') as f:
        rd = csv.reader(f)
        for row in zip(rd):
            yield row


class TrajectoryState:

    def __init__(self, episode, step, state_before, state_next, d_food, d_enemy, feats, action, reward):
        """
        A state of the expert's trajectory.
        :param episode: The episode of the algorithm in which this trajectory step produced.
        :param step: The step of the algorithm in which this trajectory step produced.
        :param state_before: State before.
        :param state_next: The state visited after the action took place.
        :param d_food: manhattan distance between agent and food.
        :param d_enemy: manhattan distance between agent and enemy.
        :param feats: Features.
        :param action: Selected action.
        :param reward: Awarded reward by moving to the current state.
        """
        self.episode = episode
        self.step = step
        self.state_before = state_before
        self.state_next = state_next
        self.d_food = d_food
        self.d_enemy = d_enemy
        self.feats = feats
        self.action = action
        self.reward = reward


def csvdata_to_trajectory_state(csv_entry):
    """
    Map csv data to a trajectory state instance.
    :param csv_entry: csv data
    :return: a trajectory instance
    """
    tr = TrajectoryState(
        episode=int(csv_entry[0][0]),
        step=int(csv_entry[0][1]),
        state_before=GridState(
            player_position=(int(csv_entry[0][2]), int(csv_entry[0][3])),
            food_position=(int(csv_entry[0][4]), int(csv_entry[0][5])),
            enemy_position=(int(csv_entry[0][6]), int(csv_entry[0][7]))
        ),
        state_next=GridState(
            player_position=(int(csv_entry[0][8]), int(csv_entry[0][9])),
            food_position=(int(csv_entry[0][10]), int(csv_entry[0][11])),
            enemy_position=(int(csv_entry[0][12]), int(csv_entry[0][13]))
        ),
        d_food=float(csv_entry[0][14]),
        d_enemy=float(csv_entry[0][15]),
        feats=ast.literal_eval(csv_entry[0][16]),
        action=int(csv_entry[0][17]),
        reward=float(csv_entry[0][18]),
    )
    return tr


def process_trajectories(expert_file: str, env, m, weak_estimation, save_files=False):
    """
    This function process the trajectories from a file and constructs feature and transition matrices
    :param expert_file: Name of the expert file containing trajectories used to teach the apprentice.
    :param env: The environment.
    :param m: Total number of trajectories from the expert.
    :param weak_estimation: Determines whether we have the number of expert trajectories required from the algorithm.
    :return:
    """
    process_start = datetime.datetime.utcnow()
    nS = env.state_space_size
    nA = env.action_space_size

    total_steps = m * env.episode_steps
    te = ThetaEstimator(env=env, m=m, weak_estimation=weak_estimation, fname=expert_file, modify_env=True)

    for step in tqdm(trajectory_generator(expert_file), total=total_steps, desc="Processing expert trajectories"):

        t = csvdata_to_trajectory_state(step)

        state_index = env.state_space_index[t.state_before.__str__()]
        next_state_index = env.state_space_index[t.state_next.__str__()]

        sa = GridState(player_position=t.state_before.pp,
                       food_position=t.state_before.fp,
                       enemy_position=t.state_before.ep,
                       action=t.action)
        sa_index = env.action_state_index[sa.__str__()]

        te.add_transition(sa=sa_index, st_next=next_state_index, st=state_index, feats=t.feats, step=t.step)
    return te.Fz, te.TRz, te.E
