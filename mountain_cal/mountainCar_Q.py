import gym
import numpy as np
from vizualization import qTableUi


class MountainCar_Q():
    global Q

    def __init__(self, alpha: float = 0.8, gamma: float = 0.95, episodes: int = 1000, silent: bool = False,
                 resume_Q: bool = False, save_Q: bool = True, checkpoint_name: str = "_Checkpoint",
                 tableUI: bool = True):
        self._alpha = alpha  # learning rate
        self._gamma = gamma  # discount factor
        self._episodes = episodes  # training episodes
        self._checkpoint_name = checkpoint_name  # checkpoint file name to save or retrieve
        self._silent = silent  # print statistics about the running algorithm progress
        self._tableUI = tableUI  # show a graphic visualization of Q table

        '''
        Environment Notes:
        
             Observations come in a two-dim vector (p, v)
             p = Car position: -1.2 <= p <= 0.6
             v = Car velocity: -0.07 <= v <= 0.07
            
            Action space is [0, 1, 2] where 0=left, 1=nothing, 2=right
            In the non-continuous MountainCar environment, the car is given a fixed impulse in the chosen direction.
        '''
        self._env = gym.make('MountainCar-v0')

        self.a_N = self._env.action_space.n  # Number of actions
        self.s_velocitybins = [round(-0.07 + 0.01 * i, 2) for i in range(0, 14)]  # Seperate velocity range in 14 bins.
        self.s_positionbins = [round(-1.2 + 0.1 * i, 1) for i in range(0, 18)]  # Separate position range in 18 bins.
        self.s_N = len(self.s_velocitybins) * len(self.s_positionbins)  # Number of states

        if resume_Q:
            self.Q = self._loadQ()  # Load Q-table to resume training
        else:
            self.Q = np.zeros((self.s_N, self.a_N))  # Initialize Q-table

    def _saveQ(self, txtCopy=False):
        """
         Saves a copy of the Q-table in numpy's .npy format, and optionally in .txt format.
        """
        # Saves a copy of the Q-table in numpy's .npy format, and optionally in .txt format.
        path = r"/media/Users/p.moraitis.UBUNTU/projects/thesis/pyMwal/checkpoints/"
        np.save(path + "MC_Qtable" + self._checkpoint_name + ".npy", self.Q)
        np.savetxt(path + "MC_Qtable" + self._checkpoint_name + ".txt", self.Q)

    def _loadQ(self):
        """
        Loads a save Q-table.
        """
        path = r"/media/Users/p.moraitis.UBUNTU/projects/thesis/pyMwal/checkpoints/"
        return np.load(path + "MC_Qtable" + self._checkpoint_name + ".npy")

    def _printif(self, string):
        if not self._silent: print(string)

    def _getRowIndex(self, state):
        s_positionbins_ext = self.s_positionbins + [0.60001]  # add right endpoint to bins to help loop indexing
        s_velocitybins_ext = self.s_velocitybins + [0.070001]
        # States are indexed by position first, then velocity. So all states with position -1.2 <= p < -1.1 would occupy the first k indices.
        for i in range(len(self.s_positionbins)):
            if (state[0] >= s_positionbins_ext[i]) and (state[0] < s_positionbins_ext[i + 1]):  # Position

                # Once the position bin is found and stored in i, move on to find velocity bin.
                for j in range(len(self.s_velocitybins)):
                    if (state[1] >= s_velocitybins_ext[j]) and (state[1] < s_velocitybins_ext[j + 1]):  # Velocity
                        return i * len(self.s_velocitybins) + j

                raise RuntimeError("Velocity state outside bounds of s_velocitybins.")
                return -1

        raise RuntimeError("Position state outside bounds of s_positionbins.")
        return -1

    def run(self):
        """
            Q-Learning Algorithm
        """
        if self._tableUI:
            qT = qTableUi.QTableUi(self.s_N, self.a_N, )
            qT.tableUpdated = False
            qT.create_Qtable()

        for ep in range(self._episodes):

            ## Episode start ##
            obsv = self._env.reset()  # reset the environment and store initial state
            self._printif("====+====+====+====+====+====+====+====+====")
            self._printif("|-- Episode {} starting.".format(ep + 1))

            t = 0
            total_reward = 0

            while True:
                self._env.render()

                ## Choose an action (with noise) ##
                noise = np.random.randn(1, self.a_N) * (1. / (ep + 1)) ** (
                    0.75)  # Generate a random noise distribution for the 3-dim action vector that attenuates as episodes go on
                action = np.argmax(self.Q[self._getRowIndex(obsv),
                                   :] + noise)  # This noise causes exploration, so our algorithm will explore less over time

                last_obsv = obsv
                obsv, reward, done, _ = self._env.step(action)  # Feed action into env and observe result
                t += 1
                total_reward += reward

                # Update Q-table
                self.Q[self._getRowIndex(last_obsv), action] = (1 - self._alpha) * self.Q[
                    self._getRowIndex(last_obsv), action] + self._alpha * (
                                                                       reward + self._gamma * np.max(
                                                                   self.Q[self._getRowIndex(obsv), :]))
                if self._tableUI:
                    qT.tableUpdated = True

                if ((t % 50 == 0 and t > 1) or done):
                    self._printif("| [t = {}]\tReward = {:.4f}".format(t, total_reward))

                if done:
                    if (t < 200):
                        success = "\t\t< Success! >"
                    else:
                        success = ""

                    self._printif("|-- Episode {} finished after {} timesteps.".format(ep + 1, t) + success)
                    break

            if ((ep + 1) % 25 == 0):
                self._saveQ(txtCopy=True)
                self._printif("|-- Q-table checkpoint saved.".format(t, total_reward))


if __name__ == '__main__':
    car = MountainCar_Q(tableUI=False, checkpoint_name="_Checkpoint", resume_Q=True)
    car.run()
