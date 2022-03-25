import sys
import time
import msvcrt
from threading import Thread

from game import Goes
from com_moudles import asyncio

next_game_prompt = '输入回车开始下一局游戏，按ESC退出'
ensure_next_game = True

def main(gameMain:Goes):
    gameMain.train(int(1e5), True)
    # gameMain.antagonist(50)

    global ensure_next_game
    while True:
        if ensure_next_game:
            ensure_next_game = False
            gameMain.start()
            print(next_game_prompt)

def input_system():
    global ensure_next_game
    while True:
        if msvcrt.kbhit():
            get_input = msvcrt.getch()
            if get_input == b'\x1b':
                sys.exit(0)
            if get_input == b'\r':
                ensure_next_game = True
                continue

if __name__ == "__main__":
    # gSize = input('Init boundary size(ex: ROWS COLS):')
    # gSize = tuple(map(int, gSize.split()))
    # gSize = max(gSize)
    # gameMain = Goes((gSize, gSize))

    gSize = (3,3)
    gameMain = Goes(gSize, train=True)
   
    tasks = [gameMain.load(True)]
    
    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.wait(tasks))
    
    main_thread = Thread(target=main, args=[gameMain])
    main_thread.setDaemon(True)
    main_thread.start()

    input_thread = Thread(target=input_system)
    input_thread.start()

    main_thread.join()
    input_thread.join()
