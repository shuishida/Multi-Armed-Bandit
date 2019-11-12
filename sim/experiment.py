import pickle
from datetime import datetime
import numpy as np


class Experiment(object):
    def __init__(self, envs, bandit, name, args, save=True):
        self.name = name
        self.envs = envs
        self.bandit = bandit
        self.type_bandit = args.bandit
        self.eval_regrets = args.regrets
        self.n_regret_eval = args.n_regret_eval
        self.n_steps = args.n_steps
        self.n_runs = args.n_runs
        self.labels = [env.agent.name for env in envs]
        self.results = None
        self.entries = ["actions", "rewards", "cum_rewards", "cum_mean", "regrets", "final_regrets"]
        self.run()
        if save:
            self.save()

    def run(self):
        n_envs = len(self.envs)
        n_actions = self.bandit.n_actions
        n_steps = self.n_steps
        n_runs = self.n_runs

        n_reg = self.n_regret_eval
        Ts = np.linspace(0, n_steps, n_reg + 1).astype(int)

        rewards_runs = np.zeros((n_envs, n_steps))
        cum_rewards_runs = np.zeros((n_envs, n_steps))
        cum_rewards_mean_runs = np.zeros((n_envs, n_steps))
        regrets_runs = np.zeros((n_envs, n_steps))
        cum_actions_count_runs = np.zeros((n_envs, n_steps, n_actions))
        final_regrets = None

        for i_env, env in enumerate(self.envs):
            for i_run in range(self.n_runs):
                actions, rewards, cum_rewards, cum_rewards_mean, regrets = env.run(n_steps=self.n_steps)

                rewards_runs[i_env] += np.array(rewards)
                cum_rewards_runs[i_env] += np.array(cum_rewards)
                cum_rewards_mean_runs[i_env] += np.array(cum_rewards_mean)
                regrets_runs[i_env] += np.array(regrets)
                cum_actions_count_runs[i_env, np.arange(n_steps), np.array(actions)] += 1.0

        rewards_runs /= n_runs
        cum_rewards_runs /= n_runs
        cum_rewards_mean_runs /= n_runs
        regrets_runs /= n_runs

        if self.eval_regrets:
            final_regrets = np.zeros((n_envs, len(Ts)))
            final_regrets[:, -1] = regrets_runs[:, -1]
            for i, T in enumerate(Ts):
                if i == 0 or T == n_steps:
                    continue
                final_regrets_sum = np.zeros(n_envs)
                for i_env, env in enumerate(self.envs):
                    for i_run in range(self.n_runs):
                        _, _, _, _, regrets = env.run(n_steps=T)
                        final_regrets_sum[i_env] += regrets[-1]
                final_regrets[:, i] = final_regrets_sum / n_runs

        self.results = {
            "actions": cum_actions_count_runs,
            "rewards": rewards_runs,
            "cum_rewards": cum_rewards_runs,
            "cum_mean": cum_rewards_mean_runs,
            "regrets": regrets_runs,
            "final_regrets": final_regrets
        }

    def get_results(self):
        if self.results is None:
            return [None for _ in self.entries]
        return [self.results[e] for e in self.entries]

    def save(self):
        now = datetime.now()
        current_time = now.strftime("%y%m%d_%H%M%S")
        filename = "{}_{}_{}_{}_steps_{}_runs".format(current_time, self.type_bandit, self.name, self.n_steps, self.n_runs)
        filepath = "data/" + filename + ".p"

        with open(filepath, 'wb') as handle:
            pickle.dump(self, handle)
