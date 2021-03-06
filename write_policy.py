import os
import numpy as np

def write_out_policies(policies, expert_file):
    print("Writing out policies..")
    expert_file = os.path.splitext(expert_file)[0]
    dir = os.path.join('saved_files', expert_file, 'policies')
    if not os.path.exists(dir):
        os.mkdir(dir)
    for i, p in enumerate(policies):  # type: (int, np.ndarray)
        file = os.path.join(dir, 'policy_' + str(i) + '.csv')
        np.savetxt(file, p, delimiter=",", fmt="%d")
    print("Done")
    return dir
