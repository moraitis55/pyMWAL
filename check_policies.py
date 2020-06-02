import csv
import os

DIR = os.path.join('saved_files', '10x50000x200x__re(25, -300, -1)__pass4__avg__4.41__success rate__0.97286_episodes_collected2500', 'policies_VVzero')
DIR2 = os.path.join('saved_files', 'dizzy40%-Truex50000pass4__avg__-147.17__success rate__0.97826_episodes_collected2500', 'policies500')

def print_policy_spreading(dir, policies=(1,2)):
    file1 = 'policy_' + str(policies[0]) + '.csv'
    file2 = 'policy_' + str(policies[1]) + '.csv'
    t1 = open(os.path.join(dir, file1), 'r')
    t2 = open(os.path.join(dir, file2), 'r')
    fileone = t1.readlines()
    filetwo = t2.readlines()
    t1.close()
    t2.close()

    differences =0
    x=0
    for i in fileone:
        if i != filetwo[x]:
            differences +=1
        x+=1
    diff_perc = (differences/x)*100
    print("These two policies are {0}% different\n{1} differences found".format(diff_perc, differences))

print_policy_spreading(DIR2, (1, 461))