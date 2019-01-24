import numpy as np
from SMPyBandits.Policies import EpsilonGreedy, EpsilonDecreasing, UCB, UCBalpha, klUCB, BayesUCB
from scipy.stats import beta
from math import sqrt, log

class ClassicEpsilonGreedy(EpsilonGreedy):

    def __str__(self):
        return f"Epsilon-greedy($\epsilon={self._epsilon}$)"
    
    def __init__(self, nbArms, epsilon=0.1, lower=0., amplitude=1.):
        super(ClassicEpsilonGreedy, self).__init__(nbArms, epsilon=epsilon, lower=lower, amplitude=amplitude)
  
    def choice(self):
        # Generate random number
        p = np.random.rand()
        """With a probability of epsilon, explore (uniform choice), otherwise exploit based on empirical mean rewards."""
        if p < self.epsilon: # Proba epsilon : explore
            #return np.random.randint(0, self.nbArms - 1)
            return np.random.randint(0, self.nbArms)
        else:  # Proba 1 - epsilon : exploit
            # Uniform choice among the best arms
            #biased_means = self.rewards / (1 + self.pulls)
            estimated_means = self.rewards / np.maximum(1, self.pulls)
            return np.random.choice(np.flatnonzero(estimated_means == np.max(estimated_means)))

class ClassicEpsilonDecreasing(ClassicEpsilonGreedy):
    r""" The epsilon-decreasing random policy.

    - :math:`\varepsilon(t) = \min(1, \varepsilon_0 / \max(1, t))`
    - Ref: https://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies
    """

    def __init__(self, nbArms, epsilon=1.0, lower=0., amplitude=1.):
        super(ClassicEpsilonDecreasing, self).__init__(nbArms, epsilon=epsilon, lower=lower, amplitude=amplitude)

    def __str__(self):
        return f"EpsilonDecreasing({self._epsilon})"

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/2/library/functions.html#property
    @property
    def epsilon(self):
        r"""Decreasing :math:`\varepsilon(t) = \min(1, \varepsilon_0 / \max(1, t))`."""
        return min(1, self._epsilon / max(1, self.t))

class ClassicOptimisticGreedy(ClassicEpsilonGreedy):
    
    def __init__(self, nbArms, epsilon=0.0, init_estimation=10.0, lower=0., amplitude=1.):
        super(ClassicEpsilonGreedy, self).__init__(nbArms, epsilon=epsilon, lower=lower, amplitude=amplitude)
        #self.estimated_means = np.repeat(init_estimation, nbArms)
        self.init_estimation = init_estimation

    def __str__(self):
        return f"OptimisticGreedy({self.init_estimation})"
        
    def choice(self):
        # Generate random number
        p = np.random.rand()
        """With a probability of epsilon, explore (uniform choice), otherwhise exploit based on empirical mean rewards."""
        if p < self.epsilon: # Proba epsilon : explore
            #return np.random.randint(0, self.nbArms - 1)
            return np.random.randint(0, self.nbArms)
        else:  # Proba 1 - epsilon : exploit
            # Uniform choice among the best arms
            estimated_means = (self.rewards + self.init_estimation) / (self.pulls + 1)
            return np.random.choice(np.flatnonzero(estimated_means == np.max(estimated_means)))

class SafeAlg:

    def __init__(self, nbArms, inibudget=10.0, safebudget=1.0):
        self.inibudget=inibudget
        self.safebudget=safebudget
        self.totalreward=0.0
        self.budget=inibudget
        self.estmeans = np.zeros(nbArms)

    def startGame(self):
        self.totalreward = 0.0
        self.budget=self.inibudget
        self.estmeans.fill(0.0)

    def getReward(self, arm, reward):
        self.totalreward += reward
        self.budget += reward
        self.estmeans[arm] = (self.estmeans[arm] * (self.pulls[arm]-1) + reward) / self.pulls[arm]

    def choice(self):
        #sufficient budget
        if self.budget > self.safebudget:
            return None
        #low budget
        else:
            if np.max(self.estmeans) > 0:
                # Uniform choice among the best arms
                return np.random.choice(np.flatnonzero(self.estmeans == np.max(self.estmeans)))
            else:
                return None

class SafeUCB(UCB, SafeAlg):

    def __str__(self):
        return f"Safe-UCB($b_s={self.safebudget}$)"

    def __init__(self, nbArms, inibudget=10.0, safebudget=1.0, lower=-1.0, amplitude=2.0):
        UCB.__init__(self, nbArms, lower=lower, amplitude=amplitude)
        SafeAlg.__init__(self, nbArms)

    def startGame(self):
        UCB.startGame(self)
        SafeAlg.startGame(self)

    def getReward(self, arm, reward):
        UCB.getReward(self, arm, reward)
        SafeAlg.getReward(self, arm, reward)

    def choice(self):
        r = SafeAlg.choice(self)
        if r is None:
            r = UCB.choice(self)
        return r

class SafeKLUCB(klUCB, SafeAlg):

    def __str__(self):
        return f"Safe-KL-UCB($b_s={self.safebudget}$)"

    def __init__(self, nbArms, inibudget=10.0, safebudget=1.0, lower=-1.0, amplitude=2.0):
        klUCB.__init__(self, nbArms, lower=lower, amplitude=amplitude)
        SafeAlg.__init__(self, nbArms)

    def startGame(self):
        klUCB.startGame(self)
        SafeAlg.startGame(self)

    def getReward(self, arm, reward):
        klUCB.getReward(self, arm, reward)
        SafeAlg.getReward(self, arm, reward)

    def choice(self):
        r = SafeAlg.choice(self)
        if r is None:
            r = klUCB.choice(self)
        return r

class SafeUCBalpha(UCBalpha, SafeAlg):

    def __str__(self):
        return f"Safe-UCB($a={self.alpha}, b_s={self.safebudget}$)"
        #return r"UCB($\alpha={:.3g}$)".format(self.alpha)

    def __init__(self, nbArms, alpha=4.0, inibudget=10.0, safebudget=1.0, lower=-1.0, amplitude=2.0):
        UCBalpha.__init__(self, nbArms, alpha=alpha, lower=lower, amplitude=amplitude)
        SafeAlg.__init__(self, nbArms)

    def startGame(self):
        UCBalpha.startGame(self)
        SafeAlg.startGame(self)

    def getReward(self, arm, reward):
        UCBalpha.getReward(self, arm, reward)
        SafeAlg.getReward(self, arm, reward)

    def choice(self):
        r = SafeAlg.choice(self)
        if r is None:
            r = UCBalpha.choice(self)
        return r

class SafeEpsilonGreedy(ClassicEpsilonGreedy, SafeAlg):

    def __str__(self):
        return f"Safe-$\epsilon$-greedy($\epsilon={self._epsilon}, b_s={self.safebudget}$)"

    def __init__(self, nbArms, epsilon=0.1, inibudget=10.0, safebudget=1.0, lower=-1.0, amplitude=2.0):
        ClassicEpsilonGreedy.__init__(self, nbArms, epsilon=epsilon, lower=lower, amplitude=amplitude)
        SafeAlg.__init__(self, nbArms)

    def startGame(self):
        ClassicEpsilonGreedy.startGame(self)
        SafeAlg.startGame(self)

    def getReward(self, arm, reward):
        ClassicEpsilonGreedy.getReward(self, arm, reward)
        SafeAlg.getReward(self, arm, reward)

    def choice(self):
        r = SafeAlg.choice(self)
        if r is None:
            r = ClassicEpsilonGreedy.choice(self)
        return r

    
    
class GamblerBayesUCB(BayesUCB):
    """ The Bayes-UCB policy, replacing $t$ by $b$.

    - By default, it uses a Beta posterior (:class:`Policies.Posterior.Beta`), one by arm.
    -Reference: [Kaufmann, Cappé & Garivier - AISTATS, 2012].
    """

    def __init__(self, nbArms, inibudget=10.0, lower=-1.0, amplitude=2.0):
        BayesUCB.__init__(self, nbArms, lower=lower, amplitude=amplitude)
        self.inibudget=inibudget
        self.budget=inibudget

    def startGame(self):
        BayesUCB.startGame(self)
        self.budget=self.inibudget

    def getReward(self, arm, reward):
        UCB.getReward(self, arm, reward)
        self.budget += reward

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k, giving :math:`S_k(t)` rewards of 1, by taking the :math:`1 - \frac{1}{\min(b, 2)}` quantile from the Beta posterior:

        .. math:: I_k(t) = \mathrm{Quantile}\left(\mathrm{Beta}(1 + S_k(t), 1 + N_k(t) - S_k(t)), 1 - \frac{1}{\min(b,2)}\right).
        """
        return self.posterior[arm].quantile(1. - 1. / (1 + min2, self.budget))
    

    
class PositiveGamblerUCB(UCB):
    
    def __init__(self, nbArms, inibudget=10.0, lower=-1.0, amplitude=2.0):
        UCB.__init__(self, nbArms, lower=lower, amplitude=amplitude)
        self.inibudget=inibudget
        self.budget=inibudget
        self.estmeans = np.zeros(nbArms)
        self.successes = np.zeros(nbArms, dtype='int')

    def startGame(self):
        UCB.startGame(self)
        self.budget=self.inibudget
        self.estmeans.fill(0.0)
        self.successes.fill(0)

    def getReward(self, arm, reward):
        UCB.getReward(self, arm, reward)
        self.budget += reward
        self.estmeans[arm] = (self.estmeans[arm] * (self.pulls[arm]-1) + reward) / self.pulls[arm]
        if(reward > 0):
            self.successes[arm] += 1
            
    def computeIndex(self, arm):
        
        if self.pulls[arm] < 1:
        
            return float('+inf')
        
        else:
        
            #v = self.estmeans[arm]  #UCB1
            #v = self.rewards[arm] / self.pulls[arm]  #UCB1
            
            #PositiveHoeffdingGambler
            v = 1/2 * np.exp(-(2.0 * self.pulls[arm]**2 * self.estmeans[arm]**2) / (self.pulls[arm] * self.amplitude**2)) 
            if (self.estmeans[arm] > 0) :
                v = 1-v
            
            #v = beta.cdf(0.5, self.pulls[arm]-self.successes[arm]+1, self.successes[arm]+1)  #PositiveBernoulliGambler
            
            #u = sqrt((2 * max(1, log(self.t))) / self.pulls[arm])   #UCB1 
            u = sqrt((2 * log(self.budget)) / self.pulls[arm]) if (self.budget >= 1) else 0
            
            return  v + u 

    def computeAllIndex(self):
        for arm in range(self.nbArms):
            self.index[arm] = self.computeIndex(arm)
            
            
            
    
class GamblerBernoulliUCB(UCB):
    
    def __init__(self, nbArms, inibudget=10.0, lower=-1.0, amplitude=2.0):
        UCB.__init__(self, nbArms, lower=lower, amplitude=amplitude)
        self.inibudget=inibudget
        self.budget=inibudget
        self.estmeans = np.zeros(nbArms)
        self.successes = np.zeros(nbArms, dtype='int')

    def startGame(self):
        UCB.startGame(self)
        self.budget=self.inibudget
        self.estmeans.fill(0.0)
        self.successes.fill(0)

    def getReward(self, arm, reward):
        UCB.getReward(self, arm, reward)
        self.budget += reward
        self.estmeans[arm] = (self.estmeans[arm] * (self.pulls[arm]-1) + reward) / self.pulls[arm]
        if(reward > 0):
            self.successes[arm] += 1
            
    def computeIndex(self, arm):
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            #v = self.estmeans[arm]
            #v = self.rewards[arm] / self.pulls[arm]
            v = beta.cdf(0.5, self.pulls[arm]-self.successes[arm]+1, self.successes[arm]+1)
            #u = sqrt((2 * max(1, log(self.t))) / self.pulls[arm])
            u = sqrt((2 * log(self.budget)) / self.pulls[arm]) if (self.budget >= 1) else 0
            return  v + u 

    def computeAllIndex(self):
        for arm in range(self.nbArms):
            self.index[arm] = self.computeIndex(arm)