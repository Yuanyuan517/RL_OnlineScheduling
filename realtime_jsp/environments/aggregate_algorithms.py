from realtime_jsp.algorithms.sarsa import SARSA
from realtime_jsp.algorithms.Q_learning import q_learning_funcs
from realtime_jsp.algorithms.edd import EDD
from realtime_jsp.algorithms.random import Random
import numpy as np
import matplotlib.style
from configparser import ConfigParser
from realtime_jsp.environments.JSPEnv2 import JSPEnv2
from realtime_jsp.utilities.plotting import Plotting


if __name__ == '__main__':
    matplotlib.style.use('ggplot')
    plotting = Plotting()
    env = JSPEnv2()
    _conf = ConfigParser()
    res = _conf.read('/Users/yuanyuanli/PycharmProjects/RL-RealtimeScheduling/realtime_jsp'
                     '/etc/app.ini')
    print(res)
    #  Train the model
    algos = [q_learning_funcs, SARSA]
    algos2 = []#[Random, EDD]
    criterion = [1, 2, 3]
    num_episodes_trains = _conf.get('algorithms', 'num_episodes_trains').split()
    lambda_value = 1/int(_conf.get('event', 'interarrival_time'))

    obj = 1
    compare_best_setting = True

    # plotting.plot_episode_stats(stats)
    filename = '/Users/yuanyuanli/PycharmProjects/RL-RealtimeScheduling/realtime_jsp/results/' \
               'bestSettingQSarsaV3.txt'
               #'1000V3.txt'
    with open(filename, 'a') as f:
        # algo of random and EDD
        for algo2 in algos2:
            model = algo2(env, _conf)
            model.obj = obj
            stats = model.run(plotting)
            s = model.name + " " + str(0) + " "
            f.write(s)
            f.write("\n")
            b = np.matrix(stats.episode_obj)
            np.savetxt(f, b, fmt="%d")
            f.write("\n")
        # algo of RL
        #'''
        for algo in algos:
            for c in criterion:
                if compare_best_setting:
                    c = 2
                for num in num_episodes_trains:
                    env.criteria = c
                    if compare_best_setting:
                        env.interarrival_mean_time = 5 # compare scale 10, 20, 5
                    train = algo(env, _conf)
                    if compare_best_setting:
                        train.size_time_steps = 5000
                    train.obj = obj
                    train.criteria = c
                    train.num_episodes_train = int(num)
                    Q, stats = train.learn(plotting)
                    # event simulator is fixed
                    # test the model with calculated Q
                    Q2, stats2 = train.fixed_seed(Q, plotting)
                    cri = ""
                    if train.criteria == 1:
                        cri = "DD"
                    elif train.criteria == 2:
                        cri = "DD_pt"
                    else:
                        cri = "random"
                    s = train.name+" "+str(num)+" "+cri#+" "+str(lambda_value)
                    f.write(s)
                    f.write(" ")
                    b = np.matrix(stats2.episode_obj)
                    np.savetxt(f, b, fmt="%d")
                    f.write("\n")
                    print("New Stats", stats2)
                if compare_best_setting: # no need to iterate criterion loop
                    break
        #'''


        print("Finished algos")
