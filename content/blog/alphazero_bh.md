+++
title = "AlphaZero: Mastering the game of Black Hole"
date = 2026-06-01T12:42:42+05:30
tags = ["Reinforcement Learning", "Game Theory"]
description = "Implementing AlphaZero for the game of Black Hole"
+++ 




This blog is a detailed insight into one of my favourite projects of creating a superhuman player for the game of Black Hole. This was a passion-project of mine which I had been planning for some time and finally got around to implementing a few months ago. AlphaZero was a pioneering work in the field of Reinforcement Learning and hence my motivation to implement it for a game. 

The general structure of this blog will be as follow:
- A brief introduction to Reinforcement Learning
- The game of Black Hole
- Intro to AlphaZero
- Monte Carlo Tree Search 
- Implementation Details
- Experiments and Ablations
- Future Work
- Resources


### Introduction to RL
Reinforcement learning is a subfield of deep learning which focuses on teaching models(termed as agents) by making them interact with some environment. The main idea is very heuristic, the model gets conditioned to perform only those actions which are beneficial to it. 

How is this "benefit" actually defined? This is done with the help of reward functions - a function $R(s, a, s')$ that takes the current state $s$, the action $a$ taken by the agent, and the next state $s'$ , and returns the reward for that action. 
some the 

//insert RL diagram

As shown above, the agent interacts with the environment, gets some rewards, trains itself based on rewards and repeat. Now while idea is simple in theory, designing reward functions and RL traininig frameworks which cater to our needs is a challenging task.


### Rules of Black Hole
Black Hole is a strategic tile-placement game played on a triangular grid. Traditionally it has 6 rows, but to push our AI to its limits, I expanded it to a massive 9-row grid (45 empty spaces).

#### *How to Play*
- **The Setup:** Two players each receive 22 tiles numbered 1 to 22. The board starts empty.
- **The Turn:** Players take turns placing one tile onto any empty spot. You *must* play your tiles in ascending numerical order (1, then 2, then 3, up to 22). 
- **The Black Hole:** Because 44 tiles are placed onto a 45-space board, exactly one space is left empty at the end. This becomes **The Black Hole**.

#### *Scoring*
The Black Hole "sucks in" neighboring tiles. Scoring is calculated in concentric rings radiating outward from the hole (Ring 1 is immediate neighbors, Ring 2 is the next layer out, etc.).

**Your goal is to get sucked in as little as possible.** The player whose tiles in Ring 1 have the *lowest sum* wins instantly. Ties are broken by evaluating Ring 2, and so on.


### Google Deepmind's AlphaGo and AlphaZero

For most of computing history, game-playing AI was built on a deceptively simple idea: search. The logic goes like this: any game has a defined set of states and transition rules. Given enough time and processing power, you could enumerate all possible futures, assign a value to the terminal state (win/loss/draw), and trace backward to assign a value to every preceding state. The optimal strategy is simply to choose the move that leads to the highest-valued state. This is called minimax algorithm.

<!-- add game state diagram for the  -->

<figure class="my-8">
  <img src="/blogs/alphazerobh/computer-science-game-trees-alphabeta.jpg" alt="minimax tree" class="w-80% rounded-sm border border-gray-200 dark:border-gray-800/70">
  <figcaption class="text-center text-sm text-gray-600 dark:text-gray-400 mt-3"><i>The game of Tic-Tac-Toe represented as a game tree. (source: https://www.yosenspace.com/posts/computer-science-game-trees.html)</i></figcaption>
</figure>

Chess engines like Stockfish perfected this approach, checking millions of positions per second, guided by handcrafted evaluation functions tuned by human grandmasters. It worked extraordinarily well, but it required too much human intervention.

But this approach was grossly insufficient for games like Go which had a massive action space for each state, hence a branching factor much larger(~7-8x) than that of chess. And give that the search space grows by the order of branching factor raised to the power of no. Of moves in the game ($b^d$), it's computationally infeasible to find the optimal move for games like Go. 

AlphaGo was the pioneering work which solve this problem. It had a two-part insight. First, a policy network learned to predict which moves strong players would make, dramatically pruning the search tree. Second, a value network learned to estimate the probability of winning from any given board state, replacing handcrafted evaluation. Both networks were trained initially on a dataset of human expert games, and then refined through self-play using reinforcement learning. 

In October 2015, AlphaGo played its first game against the reigning three-time European Champion, Fan Hui. AlphaGo won the first ever match between an AI system and Go professional, scoring 5-0.

AlphaGo then competed against legendary Go player Lee Sedol — winner of 18 world titles, and widely considered the greatest player of that decade. AlphaGo's 4-1 victory in Seoul, South Korea, in March 2016, sent shockwaves across the globe. It demonstrated that deep learning could internalize strategic intuition that no one had successfully written down.

AlphaZero, published in 2017, took this further by stripping out the human data entirely. There were no expert games and no domain-specific heuristics - just the rules of the game and time to play itself. It achieved this with the help of a modified Monte Carlo Tree Search Algorithm paired with its Policy and Value Networks which essentially guided the exploration and search process. 

Within hours of training, AlphaZero surpassed AlphaGo; within days, it exceeded the best classical engines in Go, Chess, and Shogi simultaneously — using the same algorithm for all three!

### Classical Self-Play


### Monte Carlo Tree Search

The Monte Carlo Tree Search(MCTS) is a heauristic search algorithm which helps us find near optimal moves by partially observing the future from a particular game state. It is method that helps us effectively navigate through the decision trees of problems with very large decision spaces like Go(which has $10^{170}$ possible game states). 

MCTS works by randomly sampling the search space. It does many playouts where it randomly simulates(random decisions) the entire game to the end from its current state. The final result of each playout is then used to update the statistics of the nodes visited during that playout and helps us get an estimate of which nodes lead to better outcomes. Now, the way decisions are made before the random sampling is determined by the the objection function called Upper Confidence Bound (UCB). 

>The UCB1 formula is as follows:
$$UCB1 = \frac{W}{N} + c\sqrt{\frac{\ln{N_{parent}}}{N}}$$
where $W$ is the number of wins for a node, $N$ is the number of simulations for a node, and $c$ is the exploration parameter.  

Each MCTS Sampling Iteration involves the following steps:
1. Selection - Starting at the root(current game state) node, we compute the UCB score for all its children(outcomes of actions) and select the one with the highest score and travel to it. If multiple children have same score, we choose randomly.
2. Expansion - Once we reach a node that has not been fully expanded(has legal moves which have not been visited), we add it to the tree. 
3. Simulation - From the expanded node, we simulate the game to the end by randomly selecting moves. 
4. Backpropagation - Once we reach a terminal state, we backpropagate the result to the root node by updating the statistics of all the nodes visited during the selection and simulation steps. 






{{< mcts_stepper >}}




#### Resources
- [Project Repository](https://github.com/Utsab-2010/Black_Hole_AlphaZero)
- [Original Paper](https://arxiv.org/abs/1712.01815)
- [Acquisition of Chess Knowledge in AlphaZero](https://arxiv.org/pdf/2111.09259)
- [DeepMind's AlphaZero Page](https://deepmind.com/research/publications/alphazero)