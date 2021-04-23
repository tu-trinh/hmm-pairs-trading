import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


class ProbabilityVector:
    def __init__(self, probabilities: dict):
        states = probabilities.keys()
        probs  = probabilities.values()
        
        assert len(states) == len(probs), \
            "The probabilities must match the states."
        assert len(states) == len(set(states)), \
            "The states must be unique."
        assert abs(sum(probs) - 1.0) < 1e-12, \
            "Probabilities must sum up to 1."
        assert len(list(filter(lambda x: 0 <= x <= 1, probs))) == len(probs), \
            "Probabilities must be numbers from [0, 1] interval."
        
        self.states = sorted(probabilities)
        self.values = np.array(list(map(lambda x: 
            probabilities[x], self.states))).reshape(1, -1)
        
    @classmethod
    def initialize(cls, states: list):
        size = len(states)
        rand = np.random.rand(size) / (size**2) + 1 / size
        rand /= rand.sum(axis=0)
        return cls(dict(zip(states, rand)))

    @classmethod
    def initialize_uniform(cls, states: list):
        size = len(states)
        return cls(dict(zip(states, 1/size)))
    
    @classmethod
    def from_numpy(cls, array: np.ndarray, state: list):
        return cls(dict(zip(states, list(array))))

    @property
    def dict(self):
        return {k:v for k, v in zip(self.states, list(self.values.flatten()))}

    @property
    def df(self):
        return pd.DataFrame(self.values, columns=self.states, index=['probability'])

    def __repr__(self):
        return "P({}) = {}.".format(self.states, self.values)

    def __eq__(self, other):
        if not isinstance(other, ProbabilityVector):
            raise NotImplementedError
        if (self.states == other.states) and (self.values == other.values).all():
            return True
        return False

    def __getitem__(self, state: str) -> float:
        if state not in self.states:
            raise ValueError("Requesting unknown probability state from vector.")
        index = self.states.index(state)
        return float(self.values[0, index])

    def __mul__(self, other) -> np.ndarray:
        if isinstance(other, ProbabilityVector):
            return self.values * other.values
        elif isinstance(other, (int, float)):
            return self.values * other
        else:
            NotImplementedError

    def __rmul__(self, other) -> np.ndarray:
        return self.__mul__(other)

    def __matmul__(self, other) -> np.ndarray:
        if isinstance(other, ProbabilityMatrix):
            return self.values @ other.values

    def __truediv__(self, number) -> np.ndarray:
        if not isinstance(number, (int, float)):
            raise NotImplementedError
        x = self.values
        return x / number if number != 0 else x / (number + 1e-12)

    def argmax(self):
        index = self.values.argmax()
        return self.states[index]

class ProbabilityMatrix:
    def __init__(self, prob_vec_dict: dict):
        
        assert len(prob_vec_dict) > 1, \
            "The numebr of input probability vector must be greater than one."
        assert len(set([str(x.states) for x in prob_vec_dict.values()])) == 1, \
            "All internal states of all the vectors must be indentical."
        assert len(prob_vec_dict.keys()) == len(set(prob_vec_dict.keys())), \
            "All observables must be unique."

        self.states      = sorted(prob_vec_dict)
        self.observables = prob_vec_dict[self.states[0]].states
        self.values      = np.stack([prob_vec_dict[x].values \
                           for x in self.states]).squeeze() 

    @classmethod
    def initialize(cls, states: list, observables: list):
        size = len(states)
        rand = np.random.rand(size, len(observables)) \
             / (size**2) + 1 / size
        rand /= rand.sum(axis=1).reshape(-1, 1)
        aggr = [dict(zip(observables, rand[i, :])) for i in range(len(states))]
        pvec = [ProbabilityVector(x) for x in aggr]
        return cls(dict(zip(states, pvec)))

    @classmethod
    def initialize_uniform(cls, states: list, observables: list):
        size = len(observables)
        aggr = [dict(zip(observables, 1 / size)) for i in range(len(states))]
        pvec = [ProbabilityVector(x) for x in aggr]
        return cls(dict(zip(states, pvec)))

    @classmethod
    def from_numpy(cls, array: 
                  np.ndarray, 
                  states: list, 
                  observables: list):
        p_vecs = [ProbabilityVector(dict(zip(observables, x))) \
                  for x in array]
        return cls(dict(zip(states, p_vecs)))

    @property
    def dict(self):
        return self.df.to_dict()

    @property
    def df(self):
        return pd.DataFrame(self.values, 
               columns=self.observables, index=self.states)

    def __repr__(self):
        return "PM {} states: {} -> obs: {}.".format(
            self.values.shape, self.states, self.observables)

    def __getitem__(self, observable: str) -> np.ndarray:
        if observable not in self.observables:
            raise ValueError("Requesting unknown probability observable from the matrix.")
        index = self.observables.index(observable)
        return self.values[:, index].reshape(-1, 1)

from itertools import product
from functools import reduce


class HiddenMarkovChain:
    def __init__(self, T, E, pi):
        self.T = T  # transmission matrix A
        self.E = E  # emission matrix B
        self.pi = pi
        self.states = pi.states
        self.observables = E.observables
    
    def __repr__(self):
        return "HML states: {} -> observables: {}.".format(
            len(self.states), len(self.observables))
    
    @classmethod
    def initialize(cls, states: list, observables: list):
        T = ProbabilityMatrix.initialize(states, states)
        E = ProbabilityMatrix.initialize(states, observables)
        pi = ProbabilityVector.initialize(states)
        return cls(T, E, pi)

    @classmethod
    def initialize_uniform(cls, states: list, observables: list):
        T = ProbabilityMatrix.initialize_uniform(states, states)
        E = ProbabilityMatrix.initialize_uniform(states, observables)
        pi = ProbabilityVector.initialize_uniform(states)
        return cls(T, E, pi)
    
    def _create_all_chains(self, chain_length):
        return list(product(*(self.states,) * chain_length))
    
    def score(self, observations: list) -> float:
        def mul(x, y): return x * y
        
        score = 0
        all_chains = self._create_all_chains(len(observations))
        for idx, chain in enumerate(all_chains):
            expanded_chain = list(zip(chain, [self.T.states[0]] + list(chain)))
            expanded_obser = list(zip(observations, chain))
            
            p_observations = list(map(lambda x: self.E.df.loc[x[1], x[0]], expanded_obser))
            p_hidden_state = list(map(lambda x: self.T.df.loc[x[1], x[0]], expanded_chain))
            p_hidden_state[0] = self.pi[chain[0]]
            
            score += reduce(mul, p_observations) * reduce(mul, p_hidden_state)
        return score

class HiddenMarkovChain_FP(HiddenMarkovChain):
    def _alphas(self, observations: list) -> np.ndarray:
        alphas = np.zeros((len(observations), len(self.states)))
        alphas[0, :] = self.pi.values * self.E[observations[0]].T
        for t in range(1, len(observations)):
            alphas[t, :] = (alphas[t - 1, :].reshape(1, -1) 
                         @ self.T.values) * self.E[observations[t]].T
        return alphas
    
    def score(self, observations: list) -> float:
        alphas = self._alphas(observations)
        return float(alphas[-1].sum())

class HiddenMarkovChain_Simulation(HiddenMarkovChain):
    def run(self, length: int) -> (list, list):
        assert length >= 0, "The chain needs to be a non-negative number."
        s_history = [0] * (length + 1)
        o_history = [0] * (length + 1)
        
        prb = self.pi.values
        obs = prb @ self.E.values
        s_history[0] = np.random.choice(self.states, p=prb.flatten())
        o_history[0] = np.random.choice(self.observables, p=obs.flatten())
        
        for t in range(1, length + 1):
            prb = prb @ self.T.values
            obs = prb @ self.E.values
            s_history[t] = np.random.choice(self.states, p=prb.flatten())
            o_history[t] = np.random.choice(self.observables, p=obs.flatten())
        
        return o_history, s_history

class HiddenMarkovChain_Uncover(HiddenMarkovChain_Simulation):
    def _alphas(self, observations: list) -> np.ndarray:
        alphas = np.zeros((len(observations), len(self.states)))
        alphas[0, :] = self.pi.values * self.E[observations[0]].T
        for t in range(1, len(observations)):
            alphas[t, :] = (alphas[t - 1, :].reshape(1, -1) @ self.T.values) \
                         * self.E[observations[t]].T
        return alphas
    
    def _betas(self, observations: list) -> np.ndarray:
        betas = np.zeros((len(observations), len(self.states)))
        betas[-1, :] = 1
        for t in range(len(observations) - 2, -1, -1):
            betas[t, :] = (self.T.values @ (self.E[observations[t + 1]] \
                        * betas[t + 1, :].reshape(-1, 1))).reshape(1, -1)
        return betas
    
    def uncover(self, observations: list) -> list:
        alphas = self._alphas(observations)
        betas = self._betas(observations)
        maxargs = (alphas * betas).argmax(axis=1)
        return list(map(lambda x: self.states[x], maxargs))

class HiddenMarkovLayer(HiddenMarkovChain_Uncover):
    def _digammas(self, observations: list) -> np.ndarray:
        L, N = len(observations), len(self.states)
        digammas = np.zeros((L - 1, N, N))

        alphas = self._alphas(observations)
        betas = self._betas(observations)
        score = self.score(observations)
        for t in range(L - 1):
            P1 = (alphas[t, :].reshape(-1, 1) * self.T.values)
            P2 = self.E[observations[t + 1]].T * betas[t + 1].reshape(1, -1)
            digammas[t, :, :] = P1 * P2 / score
        return digammas

class HiddenMarkovModel:
    def __init__(self, hml: HiddenMarkovLayer):
        self.layer = hml
        self._score_init = 0
        self.score_history = []

    @classmethod
    def initialize(cls, states: list, observables: list):
        layer = HiddenMarkovLayer.initialize(states, observables)
        return cls(layer)

    @classmethod
    def initialize_uniform(cls, states: list, observables: list):
        layer = HiddenMarkovLayer.initialize_uniform(states, observables)
        return cls(layer)

    def update(self, observations: list) -> float:
        alpha = self.layer._alphas(observations)
        beta = self.layer._betas(observations)
        digamma = self.layer._digammas(observations)
        score = alpha[-1].sum()
        gamma = alpha * beta / score 

        L = len(alpha)
        obs_idx = [self.layer.observables.index(x) \
                  for x in observations]
        capture = np.zeros((L, len(self.layer.states), len(self.layer.observables)))
        for t in range(L):
            capture[t, :, obs_idx[t]] = 1.0

        pi = gamma[0]
        T = digamma.sum(axis=0) / gamma[:-1].sum(axis=0).reshape(-1, 1)
        E = (capture * gamma[:, :, np.newaxis]).sum(axis=0) / gamma.sum(axis=0).reshape(-1, 1)

        self.layer.pi = ProbabilityVector.from_numpy(pi, self.layer.states)
        self.layer.T = ProbabilityMatrix.from_numpy(T, self.layer.states, self.layer.states)
        self.layer.E = ProbabilityMatrix.from_numpy(E, self.layer.states, self.layer.observables)
            
        return score

    def train(self, observations: list, epochs: int, tol=None):
        self._score_init = 0
        self.score_history = (epochs + 1) * [0]
        early_stopping = isinstance(tol, (int, float))

        for epoch in range(1, epochs + 1):
            score = self.update(observations)
            print("Training... epoch = {} out of {}, score = {}.".format(epoch, epochs, score))
            if early_stopping and abs(self._score_init - score) / score < tol:
                print("Early stopping.")
                break
            self._score_init = score
            self.score_history[epoch] = score

datasets = ['appl', 'ba', 'cl', 'ctr', 'de', 'fb', 'fdx', 'fslr', 'jnj', 'mrk', 'mrna', 'msft', 'pfe', 'pg', 'ual', 'ulta', 'uri', 'zm']

def load_data(name):
    return pd.read_csv('stock_data_2020/%s2020.csv' % name)

def results_to_csv(df, name):
    df.to_csv(name, index_label = 'Id')

from hmmlearn import hmm
from matplotlib import cm, pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator


for stock in datasets:

    aapl = load_data(stock)

    gm = GaussianMixture(n_components = 5, random_state = 12345)

    np.random.seed(13)
    model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter = 1000)

    close = (aapl['BID'][:150] + aapl['ASK'][:150]) / 2

    openPrice = aapl['OPENPRC'][:150] # [::-1]
    fracChange = (close - openPrice) / openPrice
    fracHigh = (aapl['ASKHI'][:150] - openPrice) / openPrice
    fracLow = (openPrice - aapl['BIDLO'][:150]) / openPrice

    X = np.column_stack([fracChange, fracHigh, fracLow])

    model.fit(X)

    close = (aapl['BID'][150:200] + aapl['ASK'][150:200]) / 2

    openPrice = aapl['OPENPRC'][150:200] # [::-1]
    fracChange = (close - openPrice) / openPrice
    fracHigh = (aapl['ASKHI'][150:200] - openPrice) / openPrice
    fracLow = (openPrice - aapl['BIDLO'][150:200]) / openPrice

    X = np.column_stack([fracChange, fracHigh, fracLow])

    hidden_states = model.predict(X)
    # print('hidden states')
    # print(hidden_states)
    # print('evidence')
    # print(X)

    #   print("Transition matrix")
    # print(model.transmat_)
    # print()

    print("Means and vars of each hidden state")
    for i in range(model.n_components):
        print("{0}th hidden state".format(i))
        print("mean = ", model.means_[i])
        print("var = ", np.diag(model.covars_[i]))
        print()

    dates = np.array(aapl['date'][150:200])
    close_v = close # [::-1]
    prices = aapl['PRC'][150:200]

    df = pd.DataFrame()

    fig, axs = plt.subplots(model.n_components, sharex=True, sharey=True)
    colours = cm.rainbow(np.linspace(0, 1, model.n_components))
    for i, (ax, colour) in enumerate(zip(axs, colours)):
        # Use fancy indexing to plot data in each state.
        mask = hidden_states == i
        ax.plot_date(dates[mask], close_v[mask], ".-", c=colour)
        df_temp = close_v[mask].to_frame()
        df = df.append(df_temp)
        print('df')
        print(df)
        print(type(close_v[mask]))
        print(close_v[mask])
        ax.set_title("{0}th hidden state".format(i))

        # Format the ticks.
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_minor_locator(MonthLocator())

        ax.grid(True)

    plt.show()
    df = df.sort_index()

    name = stock + '_preds.csv'
    results_to_csv(df, name)
