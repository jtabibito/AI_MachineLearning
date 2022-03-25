from com_moudles import Tuple, List, Any
from RL import AI, PredictCondition, State, EstimateInitializer

class GoesPredictCondition(PredictCondition):
    def __init__(self, states):
        super().__init__()
        self.allStates = states
        
    def exec(self, s:State, placeholder) -> Tuple[List[Any], List[State]]:
        self._predicts.clear()
        self._states.clear()
        r,c = s.data.shape
        for i in range(r):
            for j in range(c):
                if s.data[i,j] == 0:
                    self._predicts.append((i,j))
                    self._states.append(self.allStates[s.get_hash][0].next((i,j), placeholder))
        return self._predicts, self._states

class GoesEstimateInitializer(EstimateInitializer):
    def __init__(self):
        super().__init__()

    def init(self, ais:List[AI], states:List):
        for s,end_game in states:
            for ai in ais:
                if end_game:
                    if s.winner == ai.id:
                        ai._estimates[s.get_hash] = 1
                    elif s.winner == 0:
                        ai._estimates[s.get_hash] = 0.5
                    else:
                        ai._estimates[s.get_hash] = 0
                else:
                    ai._estimates[s.get_hash] = 0.5
