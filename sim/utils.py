import numpy as np
import matplotlib.pyplot as plt
import os, pickle


def load_data(filename):
    data_path = os.path.join(os.path.dirname(__file__)[:-3], filename[2:])

    with open(data_path, 'rb') as handle:
        experiment = pickle.load(handle)

    return experiment


def plot_from_data(filename):
    if "regret" in filename:
        Ts, regrets, labels = load_data(filename)
        plot_regrets([Ts], [regrets], [labels])
        plt.show()
    else:
        experiment = load_data(filename)
        plot(experiment)


def plot_regrets(Ts_arr, final_regrets, labels):
    plt.figure()
    for Ts, final_regrets_item, label in zip(Ts_arr, final_regrets, labels):
        plt.loglog(Ts, final_regrets_item, label=label)
    plt.legend()
    plt.xlabel(r"Number of total time steps $T$")
    plt.ylabel(r"$Regret(T)$")


def plot(experiment):
    n_steps = experiment.n_steps
    n_actions = experiment.bandit.n_actions
    labels = experiment.labels

    actions, rewards, cum_rewards, cum_rewards_mean, regrets, final_regrets = experiment.get_results()

    if final_regrets is not None:
        Ts = np.linspace(0, n_steps, final_regrets.shape[1]).astype(int)
        Ts_arr = [Ts for _ in range(len(labels))]
        plot_regrets(Ts_arr, final_regrets, labels)

    plt.figure()
    for cum_rewards_mean_item, label in zip(cum_rewards_mean, labels):
        plt.plot(np.arange(n_steps), cum_rewards_mean_item, label=label)
    plt.legend()
    plt.xlabel(r"Number of time steps $t$")
    plt.ylabel(r"$\overline{Reward}(t)$")

    for actions_item, label in zip(actions, labels):
        plt.figure()
        bottom_sum = np.zeros(n_steps)
        for action in range(n_actions):
            plt.title(label)
            count = actions_item[:, action]
            plt.fill_between(np.arange(n_steps), bottom_sum, count + bottom_sum, label=str(action))
            bottom_sum += count
        plt.legend()
        plt.xlabel("Number of time steps")
        plt.ylabel("Action chosen for each time step")

    plt.show()
