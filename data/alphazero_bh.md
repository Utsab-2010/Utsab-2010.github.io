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


### Rules of Black Hole?
If you've never heard of Black Hole, don't worry—it's a brilliantly simple yet deeply strategic tile-placement game. Imagine a triangular grid floating in space. 

<!-- ![9-Layer Black Hole Board](images/bh_board_9lvl.png) -->

Traditionally, the board has 6 rows, but for the sake of pushing our AI to its limits, I expanded it to a massive 9-row triangle, creating 45 empty spaces in total. I'll dive into *why* I made this jump later in the post!

#### *The Setup and the Turn*
Here’s how a match unfolds. Two players (let's call them Red and Green) face off, and each gets an identical set of 22 tiles numbered from 1 to 22. 

The board starts completely empty. Players take turns placing exactly one tile onto any empty spot on the grid. However, you don't get to choose which tile to play! You are forced to play them in increasing numerical order: on your first turn you play your "1" tile, then your "2", all the way up to "22". Since both players are doing this simultaneously, the board quickly fills up with ascending numbers.

Because there are 45 spaces on our board, and the two players place a combined total of 44 tiles... exactly one space is left totally empty at the end of the game.

This final, lonely void becomes **The Black Hole**. 

Once the Black Hole forms, it begins "sucking in" everything around it. The game is scored in concentric rings radiating outward from the hole. The first ring consists of the immediate neighbors touching the Black Hole. The second ring is the neighbors of those neighbors, and so on.

Your goal? **You want to get sucked in as little as possible.** The player whose tiles surrounding the Black Hole have the *lowest* total sum wins the game. 

First, we tally up the tiles in Ring 1. If Red’s tiles sum to 12 and Green’s sum to 18, Red wins instantly. But what if it’s a tie? Then the gravitational pull extends to Ring 2, and we tally those up to break the tie, continuing outward until a winner is found.



#### Resources
- [Project Repository](https://github.com/Utsab-2010/Black_Hole_AlphaZero)
- [Original Paper](https://arxiv.org/abs/1712.01815)
- [Acquisition of Chess Knowledge in AlphaZero](https://arxiv.org/pdf/2111.09259)
- [DeepMind's AlphaZero Page](https://deepmind.com/research/publications/alphazero)