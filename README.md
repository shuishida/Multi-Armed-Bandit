## Multi-Armed Bandit Problem
Written by Shu Ishida

This project is developed as a part of a course work assignment to compare different bandit algorithms. 
It implements the explore and exploit algorithm, $\epsilon$-greedy, successive elimination, UCB1 and UCB2.
Implementation follows the algorithms described in *Introduction to Multi-Armed Bandits by Aleksandrs Slivkins* [https://arxiv.org/pdf/1904.07272.pdf].

#### Setup
We store experiments that have been run as pickle files. Make a directory called ```data``` to store these.
```
git clone https://github.com/c16192/Multi-Armed-Bandit.git
cd Multi-Armed-Bandit
mkdir data
```

#### How to run the experiments
```
python main.py --exp <experiment number> --bandit <type of bandit>
```
```main.py``` takes other optional arguments, which can be checked by running the following: 
```
python main.py -h
```

Experiment numbers are as follows:
0. Explore-exploit algorithm
1. Optimal explore-exploit algorithm
2. Epsilon-greedy algorithm
3. Successive elimination algorithm
4. UCB1 algorithm
5. UCB2 algorithm
6. Comparing all of the above

Types of bandits are:
- *bernoulli* (default): bandit arms have bernoulli distributed rewards
- *normal*: bandit arms have Gaussian distributed rewards
- *bernoulli periodic*: success probability of the bernoulli distribution oscillates as a sinusoid.

#### How to visualise the experiments
Once the experiments have been run, they will be stored as pickle files under the ```data``` directory. While running an experiment can take a certain amount of time, plotting these results are easy.
```
python main.py --plot .\data\<path to pickle file>.p
```
