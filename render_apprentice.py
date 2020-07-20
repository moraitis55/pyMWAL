from multiprocessing import Process
from gridworld.Q import inspect_model
from gridworld.grid import GridEnv
from exec_policies import execute_policy

if __name__ == '__main__':
    env=GridEnv()

    expert = Process(target=inspect_model, args=('/media/Users/p.moraitis.UBUNTU/projects/thesis/pyMwal/gridworld/Q_checkpoints/episodic_dizzy40%_False_50000pass4__avg__4.16__success_rate__0.97136', False, 100, 'expert', env))
    expert.start()

    ap1 = Process(target=execute_policy, args=('/media/Users/p.moraitis.UBUNTU/projects/thesis/pyMwal/saved_files/episodic_dizzy40%_False_50000pass4__avg__4.16__success_rate__0.97136_episodes_collected1000000/policies', 280, env, 100000, True, '1m', False, False, False, False, 100))
    ap1.start()

    ap2 = Process(target=execute_policy, args=('/media/Users/p.moraitis.UBUNTU/projects/thesis/pyMwal/saved_files/episodic_dizzy40%_False_50000pass4__avg__4.16__success_rate__0.97136_episodes_collected4000000/policies', 1338, env, 100000, True, '4m', False, False, False, False, 100))
    ap2.start()

    ap3 = Process(target=execute_policy, args=('/media/Users/p.moraitis.UBUNTU/projects/thesis/pyMwal/saved_files/episodic_dizzy40%_False_50000pass4__avg__4.16__success_rate__0.97136_episodes_collected12000000/policies', 570, env, 100000, True, '12m', False, False, False, False, 100))
    ap3.start()


