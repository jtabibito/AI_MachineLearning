import os
import sys

from numpy import place

from com_moudles import pickle, asyncio, np 

from RL import AI, State, clamp

from game_player import Player
from game_ai import GoesEstimateInitializer, GoesPredictCondition 

class GoesState(State):
    def __init__(self, board:tuple = (3,3)):
        super(GoesState, self).__init__()
        self.data = np.zeros(board, dtype=int)
        self.end_game = False
        self.winner = None

    @property
    def get_hash(self):
        if self.hash is None:
            self.hash = 0
            for i in self.data.reshape(self.data.shape[0] * self.data.shape[1]):
                if i == -1:
                    i = 2
                self.hash = self.hash*3 + i
        return self.hash

    def next(self, index, placeholder):
        if len(index) != 2:
            raise ValueError('Input Error!')
        r,c = clamp(index[0], max_value=self.data.shape[0]-1), clamp(index[1], max_value=self.data.shape[1]-1)
        if self.data[r,c] == 0:
            n = GoesState()
            n.data = np.copy(self.data)
            n.data[r,c] = placeholder
            return n
        else:
            return None

    def end(self):
        rows, cols = self.data.shape
        results = []
        # 检验行,列
        for i in range(rows):
            results.append(sum(self.data[i,:]))
        for i in range(cols):
            results.append(sum(self.data[:,i]))
        # 检验对角线
        results.append(0)
        for i in range(rows):
            results[-1] += self.data[i,i]
        results.append(0)
        for i in range(cols):
            results[-1] += self.data[i, cols-1-i]
        
        for e in results:
            if e == 3:
                self.winner = 1
                self.end_game = True
                return self.end_game
            elif e == -3:
                self.winner = -1
                self.end_game = True
                return self.end_game
        
        if np.sum(np.abs(self.data)) == rows*cols:
            self.winner = 0
            self.end_game = True
            return self.end_game

        self.end_game = False
        return self.end_game

class Goes:
    def __init__(self, board_size:tuple = (3,3), train=True):
        self.board_size = board_size
        self.simbols = { 'boundary_x': '-'*(board_size[0]*4 + 1), 'boundary_y': '| {} '*board_size[1] + '|', '0': ' ', '1': 'X', '-1': 'O' }
        self.train_model = train
        self.epoch = 500
        self.load_mapping()
        self.end_prompt = 'RL Goes!'

    async def load(self, save=False):
        if not self.states:
            await self._goes_mapping(save)

    async def _goes_mapping(self, save=False):
        self.states = {}
        state = GoesState()
        self.states[state.get_hash] = [state, state.end()]
        
        count,maxState = 0, self.board_size[0]**(self.board_size[0]*self.board_size[1])

        async def mapping(state:GoesState, pid):
            nonlocal count
            for i in range(self.board_size[0]):
                for j in range(self.board_size[1]):
                    if state.data[i,j] == 0:
                        next_state = state.next((i,j),pid)
                        state_hash = next_state.get_hash
                        if state_hash not in self.states.keys():
                            count += 1
                            sys.stdout.flush()
                            sys.stdout.write('mapping {}/ {}\r'.format(count,maxState))
                            sys.stdout.flush()
                            is_end = next_state.end()
                            self.states[state_hash] = [next_state, is_end]
                            if not is_end:
                                await mapping(next_state, -pid)
                await asyncio.sleep(0.0001)
            sys.stdout.write('\n')

        await mapping(state, 1)
        if save:
            self.save_mapping()

    def save_mapping(self):
        with open(''.join([os.path.dirname(__file__), '\\datas\\goes_{}{}'.format(*list(self.board_size))]), 'wb') as f:
            f.write(pickle.dumps(self.states))

    def load_mapping(self):
        filepath = ''.join([os.path.dirname(__file__), '\\datas\\goes_{}{}'.format(*list(self.board_size))])
        if os.path.exists(filepath):
            with open(''.join([os.path.dirname(__file__), '\\datas\\goes_{}{}'.format(*list(self.board_size))]), 'rb') as f:
                self.states = pickle.loads(f.read())
        else:
            self.states = None

    @property
    def istrain(self):
        return self.train_model

    @istrain.setter
    def set_train(self, train):
        self.train_model = train

    @property
    def get_epoch(self):
        return self.epoch
    
    @get_epoch.setter
    def set_epoch(self, epoch):
        self.epoch = epoch

    def train(self, epoch=500, save=False):
        if not self.istrain:
            return
        self.epoch = epoch

        self.condition = GoesPredictCondition(self.states)
        GEI = GoesEstimateInitializer()

        self.player1 = AI(1, epsilon=0.01)
        self.player2 = AI(-1, epsilon=0.01)

        GEI.init([self.player1, self.player2], self.states.values())

        for e in range(self.epoch):
            self.update()
            self.player1.reset()
            self.player2.reset()
            sys.stdout.flush()
            ratio = int((e+1)/self.epoch * 10)
            sys.stdout.write('[Epoch:{0}/{1}] {2}> accuracy1:{3:.05}, accuracy2:{4:.05}\r'.format(e+1,self.epoch,''.join(['='*ratio,'.'*(10-ratio)]), self.player1.accuracy, self.player2.accuracy))
            sys.stdout.flush()
        sys.stdout.write('\n') 
        
        if save:
            # player = max(self.player1, self.player2, key=lambda x:x.accuracy)
            # player.save(''.join([os.path.dirname(__file__), '\\datas\\ai\\goes_{}{}_ai'.format(*list(self.board_size))]))
            self.player1.save(''.join([os.path.dirname(__file__), '\\datas\\ai\\goes_{}{}_ai{}'.format(*list(self.board_size), 1)]))
            self.player2.save(''.join([os.path.dirname(__file__), '\\datas\\ai\\goes_{}{}_ai{}'.format(*list(self.board_size), 2)]))
        self.set_train = False

    def _get_player(self, player1, player2):
        while True:
            yield player1
            yield player2

    def start(self):
        order = input('Choose your order(1 or 2):')
        self.condition = GoesPredictCondition(self.states)

        if order == '1':
            self.player1 = Player(1)
            self.player2 = AI(-1, epsilon=0)
            # self.player2.load(''.join([os.path.dirname(__file__), '\\datas\\ai\\goes_{}{}_ai'.format(*list(self.board_size))]))
            self.player2.load(''.join([os.path.dirname(__file__), '\\datas\\ai\\goes_{}{}_ai{}'.format(*list(self.board_size), 2)]))
        else:
            self.player1 = AI(1, epsilon=0)
            self.player2 = Player(-1)
            # self.player1.load(''.join([os.path.dirname(__file__), '\\datas\\ai\\goes_{}{}_ai'.format(*list(self.board_size))]))
            self.player1.load(''.join([os.path.dirname(__file__), '\\datas\\ai\\goes_{}{}_ai{}'.format(*list(self.board_size), 1)]))

        self.update()

    def antagonist(self, epoch=500):
        self.condition = GoesPredictCondition(self.states)
        
        self.player1 = AI(1, epsilon=0)
        self.player2 = AI(-1, epsilon=0)

        # self.player1.load(''.join([os.path.dirname(__file__), '\\datas\\ai\\goes_{}{}_ai'.format(self.player1.id,*list(self.board_size))]))
        # self.player2.load(''.join([os.path.dirname(__file__), '\\datas\\ai\\goes_{}{}_ai'.format(self.player1.id,*list(self.board_size))]))
        self.player1.load(''.join([os.path.dirname(__file__), '\\datas\\ai\\goes_{}{}_ai{}'.format(self.player1.id,*list(self.board_size), 1)]))
        self.player2.load(''.join([os.path.dirname(__file__), '\\datas\\ai\\goes_{}{}_ai{}'.format(self.player1.id,*list(self.board_size), 2)]))

        for i in range(epoch):
            self.update()
            self.player1.reset()
            self.player2.reset()

    def update(self):
        self.state = GoesState()
        self.player1.set_state(self.state, False)
        self.player2.set_state(self.state, False)
        player = self._get_player(self.player1, self.player2)
        
        next = None
        while True:
            if not self.istrain:
                self.draw()
            playing = player.send(None)
            
            if isinstance(playing, Player):
                while True:
                    uinput = playing.input()
                    next = self.state.next(uinput, playing.id)
                    if next:
                        break
                    else:
                        print('此处不能落子!')
            elif isinstance(playing, AI):
                uinput = playing.predict(self.condition)
                next = self.state.next(uinput, playing.id)

            self.state, isend = self.states[next.get_hash]
            self.player1.set_state(self.state, True)
            self.player2.set_state(self.state, True)

            if isend:
                if self.state.winner == 0:
                    self.end_prompt = '平局!'
                elif self.state.winner == 1:
                    self.end_prompt = '玩家1获胜!'
                    if isinstance(self.player1, AI):
                        self.player1.success_one()
                elif self.state.winner == -1:
                    self.end_prompt = '玩家2获胜!'
                    if isinstance(self.player2, AI):
                        self.player2.success_one()

                if self.istrain:
                    if isinstance(self.player1, AI):
                        self.player1.train()
                    if isinstance(self.player2, AI):
                        self.player2.train()
                
                if not self.istrain:
                    self.draw()
                    print(self.end_prompt)
                break

    def draw(self):
        data = self.state.data
        print(self.simbols['boundary_x'])
        for i in range(self.board_size[0]):
            print(self.simbols['boundary_y'].format(*data[i]).replace('0', self.simbols['0']).replace('-1', self.simbols['-1']).replace('1', self.simbols['1']))
            print(self.simbols['boundary_x'])
