"""
  Run this file at first, in order to see what is it printing. Instead of the print() use the respective log level
"""
############################### LOGGER
from abc import ABC, abstractmethod
from logs import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv

logging.basicConfig
logger = logging.getLogger("MAB Application")


Bandit_Reward = [1, 2, 3, 4]
NumberOfTrials = 20000
EPS = 0.1
TAU = 1 / 3



# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

ch.setFormatter(CustomFormatter())

logger.addHandler(ch)



class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass


#--------------------------------------#


class Visualization():
    """
    A class for visualizing bandit algorithms results.
    """

    def plot1(self, result, reward=True):
        """
        Plot the performance of each bandit.

        Parameters:
        - result (dict): A dictionary containing the results of the bandit experiment.
        - reward (bool): If True, plot cumulative rewards; otherwise, plot cumulative regrets.

        Returns:
        None
        """


        fig, ax = plt.subplots()

        if reward:
            cumulative_sum = np.cumsum(result['rewards'], axis=0)
        else:
            cumulative_sum = np.cumsum(result['regrets'], axis=0)
        # shape[1] is the number of bandits
        for bandit in range(cumulative_sum.shape[1]):
            if reward:
                # take logarithm to improve the scale for plotting the experiment results
                cum_sum = np.log(cumulative_sum)[:, bandit]
            else:
                # regrets are usually on a smaller scale, so no need for log
                cum_sum = cumulative_sum[:, bandit]
            ax.plot(
                np.arange(cumulative_sum.shape[0]),
                cum_sum,
                label=result['bandits'][bandit]
            )

        ax.set_title(
            f"{'Log' if reward else ''} Bandit cumulative {'reward' if reward else 'regret'} comparison")
        plt.legend()
        plt.show()


    def plot2(self, eps_greedy_results, thompson_results, trials=NumberOfTrials):
        """
        Compare E-greedy and Thompson sampling cumulative rewards and regrets.

        Parameters:
        - eps_greedy_results (dict): Results of the Epsilon-Greedy bandit algorithm.
        - thompson_results (dict): Results of the Thompson Sampling bandit algorithm.
        - trials (int): The number of trials in the experiment.

        Returns:
        None
        """

        fig, ax = plt.subplots(nrows=2, figsize=(10, 10))
        trials = np.arange(0, trials)
        ax[0].plot(trials, np.cumsum(eps_greedy_results['rewards']),
                   label='Epsilon-Greedy')
        ax[0].plot(trials, np.cumsum(thompson_results['rewards']),
                   label='Thomson sampling')
        ax[0].set_title('Total Rewards')
        ax[0].legend()
        ax[1].plot(trials, np.cumsum(eps_greedy_results['regrets']),
                   label='Epsilon-Greedy')
        ax[1].plot(trials, np.cumsum(thompson_results['regrets']),
                   label='Thompson sampling')
        ax[1].set_title('Total Regrets')
        ax[0].legend()
        plt.show()

#--------------------------------------#

class EpsilonGreedy(Bandit):
    """
        Implementation of the EpsilonGreedy algorithm with 1/t decaying epsilon
    """

    def __init__(self, true_mean, epsilon=EPS, tau=TAU):
        self.true_mean = true_mean
        self.m = 0
        self.m_estimate = 0
        self.tau = tau
        self.N = 0
        self.eps = epsilon

    def __repr__(self):
        return f"Bandit with true mean {self.true_mean}"


    def pull(self):
        """
        Pull a randomly generated value using the true mean
        Returns
        -------
        float
        """
        return (np.random.randn() / np.sqrt(self.tau)) + self.true_mean

    def update(self, x):
        """
        Update the prior estimate of the true mean
        Parameters
        ----------
        x: float: outcome of the bandits experiment
        """
        self.N += 1
        self.m_estimate = ((self.N - 1) * self.m_estimate + x) / self.N

    def experiment(self, trials=NumberOfTrials, bandit_rewards=Bandit_Reward):
        """
        Perform experiment for estimating the bandit means
        Parameters
        ----------
        trials: int: number of trials to use in the experiment
        bandit_rewards: List[float]: list of true bandit rewards
        Returns
        -------
        total_result: dict[str,list[np.ndarray,Bandit]]:
            dictionary of the experiments results
        """
        bandits = [EpsilonGreedy(m) for m in bandit_rewards]
        rewards = np.zeros((trials, len(bandits)))
        regrets = np.zeros((trials, len(bandits)))

        for i in range(0, trials):
            if np.random.random() < EPS / (i + 1):
                j = np.random.randint(len(bandits))
            else:
                j = np.argmax([b.m_estimate for b in bandits])

            x = bandits[j].pull()
            bandits[j].update(x)
            rewards[i, j] = x
            regrets[i, j] = np.max([bandit.m for bandit in bandits]) - x

        results = {'rewards': rewards, 'regrets': regrets, 'bandits': bandits}

        return results

    def report(self, result_greedy):
        """
        Print and save a report for the Epsilon Greedy bandit algorithm.

        Parameters:
        - result_greedy (dict): Results of the Epsilon-Greedy bandit algorithm.

        Returns:
        None
        """

        print("Epsilon Greedy Performance")
        vis = Visualization()
        vis.plot1(result_greedy)
        vis.plot1(result_greedy, reward = False)
        print(f'average reward is {np.mean(np.sum(result_greedy["rewards"], axis=1))}')
        print(f'average regret is {np.mean(np.sum(result_greedy["regrets"], axis=1))}')

        # create the dataframe to save as csv
        df = pd.DataFrame({'Bandit': [], 'Reward': [], 'Algorithm': []})
        for index, bandit in enumerate(result_greedy['bandits']):
            data = pd.Series({'Bandit': bandit, 'Reward': np.sum(result_greedy['rewards'][:, index]),
                              'Algorithm': 'Epsilon Greedy'})
            df = df.append(data, ignore_index=True)

        df.to_csv('report_epsilon.csv', index=False)

#--------------------------------------#


class ThompsonSampling(Bandit):
    """
    Implementation of the Thompson Sampling
    """

    def __init__(self, true_mean, tau=TAU):
        """
        Initialize the Thompson Sampling Bandit with its true mean
        Parameters
        ----------
        true_mean: float: the true mean of the bandit in the experiment
        tau: optional float: tau parameter for the precision
        """
        self.true_mean = true_mean
        self.m = 0
        self.lambda_ = 1
        self.tau = tau
        self.N = 0

    def __repr__(self):
        return f"Return {self.true_mean}"

    def pull(self):
        """
        Pull a randomly generated value using the true mean
        Returns
        -------
        float
        """
        return (np.random.rand() / np.sqrt(self.tau)) + self.true_mean

    def update(self, x):
        """
        Update the prior bayesian estimate of the true mean and variance
        Parameters
        ----------
        x: float: outcome of the experiment
        """
        self.m = (self.tau * x + self.lambda_ * self.m) / (self.tau + self.lambda_)
        self.lambda_ += self.tau
        self.N += 1

    def sample(self):
        """
        Generate a sample for choosing the bandit to pull from
        Returns
        -------
        float
        """
        return np.random.randn() / np.sqrt(self.lambda_) + self.m

    def experiment(self, trials=NumberOfTrials, bandit_rewards=Bandit_Reward):
        """
        Perform experiment for estimating the bandit means
        Parameters
        ----------
        trials: int: number of trials to use in the experiment
        bandit_rewards: List[float]: list of true bandit rewards
        Returns
        -------
        total_result: dict[str,list[np.ndarray,Bandit]]:
            dictionary of the experiments results
        """
        bandits = [ThompsonSampling(m) for m in bandit_rewards]
        rewards = np.zeros((trials, len(bandits)))
        regrets = np.zeros((trials, len(bandits)))

        for i in range(trials):
            j = np.argmax([b.sample() for b in bandits])
            x = bandits[j].pull()
            bandits[j].update(x)
            rewards[i, j] = x
            regrets[i, j] = np.max([bandit.m for bandit in bandits]) - x

        results = {'rewards': rewards, 'regrets': regrets, 'bandits': bandits}
        return results

    def report(self, result_thompson):
        """
        Print and save a report for the Thompson Sampling bandit algorithm.

        Parameters:
        - result_thompson (dict): Results of the Thompson Sampling bandit algorithm.

        Returns:
        None
        """
        print('Thompson Sampling Performance')
        vis = Visualization()
        vis.plot1(result_thompson)
        vis.plot1(result_thompson, reward=False)
        print(f'average reward is {np.mean(np.sum(result_thompson["rewards"], axis=1))}')
        print(f'average regret is {np.mean(np.sum(result_thompson["regrets"], axis=1))}')

        # create the dataframe to save as csv
        df = pd.DataFrame({'Bandit': [], 'Reward': [], 'Algorithm': []})
        for index, bandit in enumerate(result_thompson['bandits']):
            data = pd.Series({'Bandit': bandit, 'Reward': np.sum(result_thompson['rewards'][:, index]),
                              'Algorithm': 'Thompson Sampling'})
            df = df.append(data, ignore_index=True)

        df.to_csv('report_thompson.csv', index=False)


def comparison(eps_greedy_results, thompson_sampling_results, trials=NumberOfTrials):
    """
    Compare the performances of Epsilon-Greedy and Thompson Sampling algorithms visually.

    Parameters:
    - eps_greedy_results (dict): Results of the Epsilon-Greedy bandit algorithm.
    - thompson_sampling_results (dict): Results of the Thompson Sampling bandit algorithm.
    - trials (int): The number of trials in the experiment.

    Returns:
    None
    """
    fig, ax = plt.subplots(nrows=2, figsize=(10, 10))
    trials = np.arange(0, trials)
    ax[0].plot(trials, np.cumsum(eps_greedy_results['rewards']), label='Epsilon-Greedy')
    ax[0].plot(trials, np.cumsum(thompson_sampling_results['rewards']), label='Thomson sampling')
    ax[0].set_title('Total Rewards')
    ax[0].legend()
    ax[1].plot(trials, np.cumsum(eps_greedy_results['regrets']), label='Epsilon-Greedy')
    ax[1].plot(trials, np.cumsum(thompson_sampling_results['regrets']), label='Thompson sampling')
    ax[1].set_title('Total Regrets')
    ax[0].legend()
    plt.show()


if __name__ == '__main__':
   
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")
