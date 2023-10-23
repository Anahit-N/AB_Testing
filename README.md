# AB_Testing
A/B Testing Assignment
Overview
This repository contains the implementation of A/B testing using Epsilon Greedy and Thompson Sampling algorithms. The experiment involves four advertisement options (bandits), and the goal is to design and analyze the performance of both algorithms.

Bandit Class
The Bandit class is an abstract class with necessary methods for defining a bandit. It includes methods for pulling the bandit, updating its state, running an experiment, and reporting results.

Epsilon Greedy
The EpsilonGreedy class is an implementation of the Epsilon Greedy algorithm, which includes a method for decaying epsilon over time and designing the experiment. It inherits from the Bandit class.

Thompson Sampling
The ThompsonSampling class is an implementation of the Thompson Sampling algorithm, designed with a known precision. It inherits from the Bandit class.

Reporting Visualizations
Visualize the learning process for each algorithm, showing the cumulative rewards and regrets.

CSV Report
Stores the rewards in a CSV files with columns for Bandit, Reward, and Algorithm.
