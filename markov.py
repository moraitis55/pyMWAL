import json
import pickle


class TransitionMatrix:
    def __init__(self, init_name):
        self.matrix = {}
        self.norm_matrix = None
        self.fname = 'transition_data_' + init_name if init_name else 'transition_data'
        self.curr_state = None

    def add_transition(self, prev_state, action, next_state):
        if prev_state in self.matrix.keys():
            if action in self.matrix[prev_state]:
                if next_state in self.matrix[prev_state][action]:
                    self.matrix[prev_state][action][next_state] += 1
                else:
                    self.matrix[prev_state][action][next_state] = 1
            else:
                self.matrix[prev_state][action] = {}
                self.matrix[prev_state][action][next_state] = 1

        else:
            self.matrix[prev_state] = {}
            self.matrix[prev_state][action] = {}
            self.matrix[prev_state][action][next_state] = 1

    def normalize(self):
        new_matrix = {}
        for prev_freq in self.matrix.keys():
            for action in self.matrix[prev_freq].keys():
                curr_freq_data = self.matrix[prev_freq][action]
                total = sum(curr_freq_data.values())
                new_dict = {}
                for key in curr_freq_data.keys():
                    new_dict[key] = curr_freq_data[key] / total
                if prev_freq in new_matrix.keys():
                    new_matrix[prev_freq][action] = new_dict
                else:
                    new_matrix[prev_freq] = {}
                    new_matrix[prev_freq][action] = new_dict

        self.norm_matrix = new_matrix

    # Save the data into a file
    # Returns either the file name for the frequencies,
    # or the file name for the normalized probabilities
    def save(self, norm=False):
        self.normalize()
        try:
            frequency_data = json.dumps(self.matrix)
            normalized_data = json.dumps(self.norm_matrix)
            open(self.fname + '.json', 'w').write(frequency_data)
            open(self.fname + '_norm.json', 'w').write(normalized_data)
        except TypeError:
            with open(self.fname + ".pickle", "wb") as f:
                pickle.dump(self.matrix, f)
            with open(self.fname + "_norm" + ".pickle", "wb") as f:
                pickle.dump(self.norm_matrix, f)
        if not norm:
            return self.fname + '.json'
        return self.fname + '_norm.json'

