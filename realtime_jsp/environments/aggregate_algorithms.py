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
    algos = [SARSA, q_learning_funcs]#[q_learning_funcs, SARSA]
    algos2 = [Random, EDD]
    criterion = [1, 2, 3]
    num_episodes_trains = _conf.get('algorithms', 'num_episodes_trains').split()
    #[1000, 10000, 100000]
    obj = 1

    # plotting.plot_episode_stats(stats)
    filename = '/Users/yuanyuanli/PycharmProjects/RL-RealtimeScheduling/realtime_jsp/results/100000_V2.txt'
    with open(filename, 'a') as f:
        # algo of random and EDD
        for algo2 in algos2:
            for i in range(len(criterion)-1):
                env.criteria = criterion[i]
                model = algo2(env, _conf)
                model.obj = obj
                stats = model.run(plotting)
                cri = ""
                if env.criteria == 1:
                    cri = "DD"
                elif env.criteria == 2:
                    cri = "DD_pt"
                s = model.name + " " + str(0) + " " + cri + " "
                f.write(s)
                f.write("\n")
                b = np.matrix(stats.episode_rewards)
                np.savetxt(f, b, fmt="%d")
                f.write("\n")
        # algo of RL
        #'''
        for algo in algos:
            for c in criterion:
                for num in num_episodes_trains:
                    env.criteria = c
                    train = algo(env, _conf)
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
                    s = train.name+" "+str(num)+" "+cri+" "
                    f.write(s)
                    f.write("\n")
                    b = np.matrix(stats2.episode_obj)
                    np.savetxt(f, b, fmt="%d")
                    f.write("\n")
                    print("New Stats", stats2)
        #'''


        print("Finished algos")
