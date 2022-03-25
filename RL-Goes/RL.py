import os
import sys

from ast import FunctionDef
from pyclbr import Function

from com_moudles import Tuple, List, Any
from com_moudles import np
from com_moudles import pickle

def clamp(x, min_value=0, max_value=1):
    """
    Clamp a value between a minimum and maximum value,
    default min_value=0, max_value=1.
    """              
    return min_value if x < min_value else x if x < max_value else max_value

class State:
    def __init__(self):
        self.data = None
        self.hash = None
    
    @property
    def get_hash(self):
        return self.hash

    def next(self, index:Tuple[int], placeholder:float):
        """
        继承该类的类必须实现该方法
        """
        raise NotImplementedError("next() is not implemented")
        # s = State()
        # s.data = np.copy(self.data)
        # s.data[index] = placeholder
        # return s

class PredictCondition:
    def __init__(self):
        self._predicts = list[Any]()
        self._states = list[State]()
    
    def exec(self, s:State, placeholder) -> Tuple[List[Any], List[State]]:
        """
        返回包含三部分:
        1. 当前状态所有可能的动作
        2. 当前状态所有可能的下一个状态
        """
        return self._predicts, self._states

class AI:
    def __init__(self, placeholder, reward:Function = None, gamma:float = 0.1, epsilon:float = 0.1):
        self._states = list[State]()
        self._estimates = dict()
        self._gamma = list[float]()
        
        self.placeholder = placeholder
        self.epsilon = clamp(epsilon)
        self.gamma = gamma
        self.reward = reward if reward else self.reward
        self.__train_cnt = 0
        self.__success_cnt = 0

    @property
    def id(self):
        return self.placeholder

    @property
    def get_state(self):
        return self._states

    def set_state(self, state, greedy=True):
        if not isinstance(state, State):
            raise TypeError("state must be a State object")
        self._states.append(state)
        self._gamma.append(self.gamma*greedy)

    def reset(self):
        self._states.clear()
        self._gamma.clear()
    
    @property
    def accuracy(self):
        return self.__success_cnt/self.__train_cnt

    @property
    def success(self):
        return self.__success_cnt

    def success_one(self):
        self.__success_cnt += 1

    def predict(self, condition:PredictCondition):
        if len(self._states) == 0:
            self.set_state(State(), False)
        state = self._states[-1]
        predictions, states = condition.exec(state, self.placeholder)
        
        if np.random.rand() < self.epsilon:
            i = np.random.randint(0, len(predictions))
            self._gamma[-1] *= False
            # self.set_state(states[i], True)
            return predictions[i]
        else:
            values = []
            for data, state in zip(predictions,states):
                if state.get_hash not in self._estimates.keys():
                    self._estimates[state.get_hash] = np.random.rand()
                values.append((self._estimates[state.get_hash], data, state))
            np.random.shuffle(values)
            values.sort(key=lambda x:x[0], reverse=True)
            # self.set_state(values[0][2], True)
            return values[0][1]

    def reward(self, estimate2:float, estimate1:float):
        return estimate2 - estimate1

    def estimate(self, traning=True):
        if traning:
            shash = [s.get_hash for s in self._states]
            for i in range(len(shash)-2, -1, -1):
                if shash[i] in self._estimates.keys() and shash[i+1] in self._estimates.keys():
                    self._estimates[shash[i]] += self._gamma[i]*self.reward(self._estimates[shash[i+1]], self._estimates[shash[i]])
            return self._estimates.values()
        else:
            return self._estimates.values()
    
    def train(self):
        self._back_propagate()
        self.__train_cnt += 1

    def _back_propagate(self):
        self.estimate()

    def save(self, path:str):
        with open(path, 'wb') as f:
            pickle.dump(self._estimates, f)

    def load(self, path:str) -> bool:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self._estimates = pickle.load(f)
            return True
        else:
            return False

class EstimateInitializer:
    def __init__(self):
        pass

    def init(self, ais:List[AI], states:List):
        for s in states:
            for ai in ais:
                ai._estimates[s.get_hash] = np.random.rand()
