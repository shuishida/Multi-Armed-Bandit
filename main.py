from sim.iid import *
from cli import parse_command
from sim.utils import plot_from_data
from bandit.independent import IndependentBandit
from bandit.arm.normal import NormalArm
from bandit.arm.bernoulli import BernoulliArm
from bandit.arm.bernoulli_periodic import BernoulliPeriodicArm

MEANS = [0.0, 0.1, 0.2, 0.3, 0.4]
VARS = [3.0, 2.4, 1.8, 1.2, 0.6]
P_SUCCESSES = [0.4, 0.45, 0.5, 0.55, 0.6]


def get_normal_bandit(means, vars):
    arms = [NormalArm(mean, var) for mean, var in zip(means, vars)]
    return IndependentBandit(arms)


def get_bernoulli_bandit(ps):
    arms = [BernoulliArm(p) for p in ps]
    return IndependentBandit(arms)


def get_periodic_bandit(p_min, p_max, period, n):
    arms = [BernoulliPeriodicArm(p_min, p_max, period, period * i / n) for i in range(n)]
    return IndependentBandit(arms)


def main(args):

    bandit = get_bernoulli_bandit(P_SUCCESSES)
    if args.bandit == "normal":
        bandit = get_normal_bandit(MEANS, VARS)
    elif args.bandit == "periodic":
        bandit = get_periodic_bandit(0.3, 0.7, 100, 5)

    if args.plot != "":
        plot_from_data(args.plot)

    elif args.exp == 0:
        print("Explore-exploit algorithm")
        n_explores = [0, 5, 10, 50, 100, 150]
        run_exp_exp_on_iid(bandit, n_explores, args)

    elif args.exp == 1:
        print("Optimal explore-exploit algorithm")
        run_exp_exp_opt_on_iid(bandit, args)

    elif args.exp == 2:
        print("Epsilon-greedy algorithm")
        epsilons = [1e-1, 1e-2, 1e-3, 1e-4, 0.0, None]
        run_epsilon_on_iid(bandit, epsilons, args)

    elif args.exp == 3:
        print("Successive elimination algorithm")
        run_succ_elim_on_iid(bandit, args)

    elif args.exp == 4:
        print("UCB1 algorithm")
        run_ucb1_on_iid(bandit, args)

    elif args.exp == 5:
        print("UCB2 algorithm")
        alphas = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
        run_ucb2_on_iid(bandit, alphas, args)

    elif args.exp == 6:
        print("All algorithms")
        run_all_on_iid(bandit, args)


if __name__ == '__main__':
    args = parse_command()
    main(args)
