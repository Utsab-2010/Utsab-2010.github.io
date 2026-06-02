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





#### Resources
- [Project Repository](https://github.com/Utsab-2010/Black_Hole_AlphaZero)
- [Original Paper](https://arxiv.org/abs/1712.01815)
- [Acquisition of Chess Knowledge in AlphaZero](https://arxiv.org/pdf/2111.09259)
- [DeepMind's AlphaZero Page](https://deepmind.com/research/publications/alphazero)