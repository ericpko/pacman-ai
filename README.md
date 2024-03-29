Artificial Intelligence, Pacman Game
======================================

## Intro
[The Pacman Projects](http://ai.berkeley.edu/project_overview.html) by the [University of California, Berkeley](http://berkeley.edu/).

![Animated gif pacman game](http://ai.berkeley.edu/images/pacman_game.gif)

> In this project, Pacman agent will find paths through his maze world, both to reach a particular location and to collect food efficiently. Try to build general search algorithms and apply them to Pacman scenarios.

Start a game by the command:
```
$ python pacman.py
```
You can see the list of all options and their default values via:
```
$ python pacman.py -h
```

## Search
- DFS, BFS, UCS, ASTAR, ASTAR heuristic 
```
$ python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=dfs
$ python pacman.py -l bigMaze -p SearchAgent -a fn=bfs -z .5
$ python pacman.py -l bigMaze -p SearchAgent -a fn=ucs
$ python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
```
- Corner problem, Corner heuristic
```
$ python pacman.py -l mediumCorners -p SearchAgent -a fn=bfs,prob=CornersProblem
$ python pacman.py -l mediumCorners -p AStarCornersAgent -z 0.5
```
- Eating all the dots
```
$ python pacman.py -l trickySearch -p AStarFoodSearchAgent
```


## Multi-Agent
- ReflexAgent: 
A capable reflex agent will have to consider both food locations and ghost locations to perform well.
```
$ python pacman.py --frameTime 0 -p ReflexAgent -k 2
$ python pacman.py -p ReflexAgent -l openClassic -n 10 -q
```
- MinimaxAgent: 
Write an adversarial search agent in the provided MinimaxAgent class stub in multiAgents.py
```
$ python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=4
```
- AlphaBetaAgent: 
Make a new agent that uses alpha-beta pruning to more efficiently explore the minimax tree.
```
$ python pacman.py -p AlphaBetaAgent -l trappedClassic -a depth=3 -q -n 10
```
- Expectimax: 
ExpectimaxAgent is useful for modeling probabilistic behavior of agents who may make suboptimal choices.
```
$ python pacman.py -l smallClassic -p ExpectimaxAgent -a evalFn=better -q -n 30
```
