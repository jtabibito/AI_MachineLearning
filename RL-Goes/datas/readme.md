### 井字棋游戏
#### 玩法模块(module:game)
1. Goes类:
    1. 初始化
        1. 棋盘
        1. 边界符号
        1. 状态数据
    1. 绘制
        1. 棋盘框架
        1. 数据部分
            1. n*n棋盘
                1. 保存
                2. 加载
    1. 输入
        1. 接收玩家一个二维输入摆放棋子
1. State类 -> 用于状态转移:
    1. 棋子摆放状态,实现一个data表
    1. hash,实现当前状态的hash索引,用于索引棋盘数据
    1. end,实现当前状态结束判断
    1. next,从当前棋盘推断下一个棋盘数据
1. Player类 -> 实现玩家模块:
    1. input,接收玩家输入,并返回玩家数据
    
#### 强化学习模块(module:RL)
1. AI类:
    1. 初始化
        1. 行为空间 $$ \{A_0,A_1,A_2,\dots,A_t,\dots,A_{n-1},A_n \lvert A,t \in [0,n] \} $$
        1. 状态空间 $$ \{S_0,S_1,S_2,\dots,S_t,\dots,S_{n-1},S_n \lvert S,t \in [0,n] \} $$
        1. 状态回报率
        1. 行为概率(上一个状态到下一个状态的行为概率) $$ \pi(A_t \lvert S_t)=P(A_t \lvert B_t) $$
        1. 状态奖励函数 $$ R_t $$
        1. 奖励衰减因子(γ)和综合价值函数 $$ v_\pi(s)=E_\pi(\gamma^0R_{t+1} + \gamma^1R_{t+2} + \gamma^2R_{t+3} + \gamma^3R_{t+4} + \dots \lvert S_{t+1}) $$
        1. 探索率epsilon
        1. 反向传播
    1. 数据
        1. 行为概率初始化模块(EstimateInitializer)
        1. 保存数据
        1. 加载数据