import csv
import os

DIR = os.path.join('saved_files', '10x50000x200x__re(25, -300, -1)__pass4__avg__4.41__success rate__0.97286_episodes_collected2500', 'policies_VVzero')

def print_policy_spreading(policies=(1,2)):
    file1 = 'policy ' + str(policies[0]) + '.csv'
    file2 = 'policy ' + str(policies[1]) + '.csv'
    t1 = open(os.path.join(DIR, file1), 'r')
    t2 = open(os.path.join(DIR, file2), 'r')
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
    print("These two policies_VVzero are {0}% different\n{1} differences found".format(diff_perc, differences))

print_policy_spreading((0,149))