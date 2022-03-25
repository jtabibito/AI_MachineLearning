import sys

from com_moudles import Tuple

class Player:
    def __init__(self, placeholder):
        self._prompt = 'Please input your choice: (ex. 1 2) '
        self.placeholder = placeholder

    @property
    def id(self):
        return self.placeholder

    @property
    def get_state(self):
        pass

    def set_state(self, state):
        pass

    def reset(self):
        pass

    def input(self) -> Tuple:
        sys.stdout.flush()
        msg = input(self._prompt).split()
        return tuple(map(int, msg))
