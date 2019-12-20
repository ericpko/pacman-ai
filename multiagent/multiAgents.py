# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import random

import util
from game import Agent, Directions  # noqa
from util import manhattanDistance  # noqa


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        # "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newFoodLst = newFood.asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Consider edge cases first
        # Check if this successor state wins or loses the game immediately
        if (successorGameState.isWin()):
            return float('inf')
        elif (successorGameState.isLose()):
            return float('-inf')
        elif (len(newFoodLst) == 0):       # No more food -> this move wins
            return float('inf')
        # elif (action == Directions.STOP):  # Keep him moving
        #     return float('-inf')

        # Start with an initial score
        score = successorGameState.getScore()

        for foodPos in newFoodLst:
            distanceToFood = manhattanDistance(newPos, foodPos)
            if (distanceToFood > 0):
                score += (1.0 / distanceToFood)

        # NOTE: not required, but doesn't seem to get agent to eat the capsule
        # when run on autograder q1
        # for capsulePos in successorGameState.getCapsules():
        #     distanceToCapsule = manhattanDistance(newPos, capsulePos)
        #     if (distanceToCapsule > 0):
        #         score += (1 / distanceToCapsule)

        for ghostState in newGhostStates:
            ghostPos = ghostState.getPosition()
            distanceToGhost = manhattanDistance(newPos, ghostPos)

            # Since there are less ghosts than food, we should increase the
            # 'weight' of being in proximity to a ghost
            if (distanceToGhost > 0):
                if (distanceToGhost == 1):      # extra bad position
                    score -= 1
                score -= (1.0 / distanceToGhost)
                # score -= (10 * (1 - math.exp(-0.1 * distanceToFood)))

        return score


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        return self.minimaxAction(gameState)

        # NOTE: Uncomment below to run recDFMinimax version
        # (minimaxValue, bestAction) = self.recDFMinimax(gameState, self.index, self.depth * 2)
        # return bestAction

    # NOTE: The following method is the minimax implentation from the lecture
    # slides. This method works, but is not being called. See the three methods
    # below.
    def recDFMinimax(self, gameState, agent, depth):
        """
        This is a recursive depth-first minimax implementation based off the
        lecture slides.
        Returns a tuple with the minimax value and the action that caused that
        value.
        """
        minimaxAction = Directions.STOP     # arbitrary

        if (gameState.isWin() or gameState.isLose() or depth == 0):
            return self.evaluationFunction(gameState), minimaxAction

        # Get the legal actions of this current agent
        legalActions = gameState.getLegalActions(agent)

        # Case 1: The agent/player is pacman
        if (agent == 0):
            minimaxVal = float('-inf')
            for action in legalActions:
                successor = gameState.generateSuccessor(agent, action)

                (value, act) = self.recDFMinimax(successor, 1, depth - 1)
                if (value > minimaxVal):
                    minimaxVal, minimaxAction = value, action

        # Case 2: The agent is a ghost
        else:
            minimaxVal = float('inf')
            for action in legalActions:
                successor = gameState.generateSuccessor(agent, action)

                # Check if the ghost is the last agent
                if (agent == gameState.getNumAgents() - 1):
                    (value, act) = self.recDFMinimax(successor, 0, depth - 1)
                else:
                    (value, act) = self.recDFMinimax(successor, agent + 1, depth)

                if (value < minimaxVal):
                    minimaxVal, minimaxAction = value, action

        return minimaxVal, minimaxAction

    # NOTE: The following three methods are based on the textbook implmentation
    # of minimax on page 166
    def minimaxAction(self, gameState: 'GameState') -> 'Actions':
        """
        """
        # Multiply depth by 2 since each player needs to make a move per level
        depth = self.depth * 2
        agent = self.index      # agent/player
        nextAgent = (agent + 1) % gameState.getNumAgents()  # nextAgent == 1

        # Get all the legal actions, and the resulting scores from the successor
        # states of those applied actions
        legalActions = gameState.getLegalActions(agent)
        scores = [self.minValue(gameState.generateSuccessor(agent, action), nextAgent, depth - 1) for action in legalActions]

        bestIndex = scores.index(max(scores))
        return legalActions[bestIndex]

    def maxValue(self, gameState: 'GameState', agent: int, depth: int) -> float:
        """
        """
        if (gameState.isWin() or gameState.isLose() or depth == 0):
            return self.evaluationFunction(gameState)

        maxVal = float('-inf')
        for action in gameState.getLegalActions(agent):
            successor = gameState.generateSuccessor(agent, action)

            # The next agent is always going to be ghost 1
            maxVal = max(maxVal, self.minValue(successor, 1, depth - 1))

        return maxVal

    def minValue(self, gameState: 'GameState', agent: int, depth: int) -> float:
        """
        Assume that <gameState> always corresponds to a ghost agent
        """
        if (gameState.isWin() or gameState.isLose() or depth == 0):
            return self.evaluationFunction(gameState)

        nextAgent = (agent + 1) % gameState.getNumAgents()
        minVal = float('inf')
        for action in gameState.getLegalActions(agent):
            successor = gameState.generateSuccessor(agent, action)

            # Check if we are on the last agent. Note that we call minValue
            # with the same depth in the else clause
            if agent == gameState.getNumAgents() - 1:
                minVal = min(minVal, self.maxValue(successor, nextAgent, depth - 1))
            else:
                minVal = min(minVal, self.minValue(successor, nextAgent, depth))

        return minVal


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        # return self.alphaBetaSearch(gameState)

        # NOTE: note the index 1 because <recAlphaBetaSearch> returns a tuple
        return self.recAlphaBetaSearch(gameState, self.index, self.depth * 2, float('-inf'), float('inf'))[1]

    def recAlphaBetaSearch(self, gameState, agent, depth, alpha, beta):
        """
        """
        bestAction = Directions.STOP

        if (gameState.isWin() or gameState.isLose() or depth == 0):
            return self.evaluationFunction(gameState), bestAction

        # Get all the legal actions of this agent
        legalActions = gameState.getLegalActions(agent)

        # Case 1: agent/player is the pacman
        if (agent == 0):
            bestValue = float('-inf')
            for action in legalActions:
                successor = gameState.generateSuccessor(agent, action)
                (nextValue, nextAction) = self.recAlphaBetaSearch(successor, 1, depth - 1, alpha, beta)

                if (nextValue > bestValue):
                    bestValue, bestAction = nextValue, action

                # Test if we can prune
                if (bestValue >= beta):
                    return bestValue, bestAction
                else:
                    # Update alpha
                    alpha = max(alpha, bestValue)

        # Case 2: agent is a ghost
        else:
            bestValue = float('inf')
            for action in legalActions:
                successor = gameState.generateSuccessor(agent, action)

                # Check if agent is the last ghost
                if (agent == gameState.getNumAgents() - 1):
                    nextValue, nextAction = self.recAlphaBetaSearch(successor, 0, depth - 1, alpha, beta)
                # Update the agent, but do not increase the depth
                else:
                    nextValue, nextAction = self.recAlphaBetaSearch(successor, agent + 1, depth, alpha, beta)

                # Check if we can update our best values
                if (nextValue < bestValue):
                    bestValue, bestAction = nextValue, action

                # Test if we can prune
                if (bestValue <= alpha):
                    return bestValue, action
                else:
                    # Update beta
                    beta = min(beta, bestValue)

        return bestValue, bestAction

    # NOTE: alternate implementation. Same as for minimax
    def alphaBetaSearch(self, gameState: 'GameState') -> 'Action':
        """
        """
        agent = self.index
        nextAgent = (agent + 1) % gameState.getNumAgents()
        depth = self.depth * 2

        # Get the legal actions from this agent
        legalActions = gameState.getLegalActions(agent)

        # Generate a list of values from
        scores = [self.minValue(gameState.generateSuccessor(agent, action), nextAgent, depth - 1) for action in legalActions]
        bestScore = max(scores)
        bestIndex = scores.index(bestScore)

        return legalActions[bestIndex]

    # TODO: implement for extra practice
    def minValue(self, gameState, agent, depth, alpha, beta) -> float:
        """
        """

    # TODO: implement for extra practice
    def maxValue(self, gameState, agent, depth, alpha, beta) -> float:
        """
        """


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        return self.expectimaxAction(gameState)

    def expectimaxAction(self, gameState) -> 'Actions':
        """
        """
        depth = self.depth * 2
        agent = self.index
        nextAgent = (agent + 1) % gameState.getNumAgents()

        legalActions = gameState.getLegalActions(agent)
        scores = [self.expectiValue(gameState.generateSuccessor(agent, action), nextAgent, depth - 1) for action in legalActions]

        bestScore = max(scores)
        bestIndex = scores.index(bestScore)

        return legalActions[bestIndex]

    def expectiValue(self, gameState, agent, depth) -> float:
        """
        """
        if (gameState.isWin() or gameState.isLose() or depth == 0):
            return self.evaluationFunction(gameState)

        # Get all the legal actions for this agent
        actions = gameState.getLegalActions(agent)

        expectedVal = 0
        for action in actions:
            successor = gameState.generateSuccessor(agent, action)

            if (agent == gameState.getNumAgents() - 1):
                expectedVal += self.maxValue(successor, 0, depth - 1)
            else:
                expectedVal += self.expectiValue(successor, agent + 1, depth)

        return (expectedVal / len(actions))

    def maxValue(self, gameState, agent, depth) -> float:
        """
        """
        if (gameState.isWin() or gameState.isLose() or depth == 0):
            return self.evaluationFunction(gameState)

        maxVal = float('-inf')
        for action in gameState.getLegalActions(agent):
            successor = gameState.generateSuccessor(agent, action)
            maxVal = max(maxVal, self.expectiValue(successor, 1, depth - 1))

        return maxVal




def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      The idea is similar to reflex agent's eval function. We want to consider
      pacman's distance to food and ghosts. If he is close to ghosts, then
      we should decrement the score so that being near a ghost is unfavorable.
      However, in the case when the ghosts are scared, pacman should hunt the
      ghost if he can since they grant a greater reward than food. Lastly,
      pacman's score should increase by a greater amount the closer he is to
      food (hence the reciprocal).
    """
    # Get some initial values similar to reflex agent's eval function
    myPosition = currentGameState.getPacmanPosition()
    foodLst = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    # Get the initial score
    score = currentGameState.getScore()

    # Check if this state wins or loses the game immediately.
    # NOTE: Do not return infinity in these extreme cases because pacman will
    # get stuck not eating the last piece of food since the score outcome would
    # be the same as just not moving. Thank you https://piazza.com/class/jv3btdgvnzmm1?cid=157
    if (currentGameState.isWin() or len(foodLst) == 0):
        score += 1000                        # Big reward
    elif (currentGameState.isLose()):
        score -= 1000                        # Punish undesirable state


    # Case 1: proximity to ghosts and scared ghosts
    ghostScore = scaredGhostScore = 0
    for ghostState in ghostStates:
        ghostPos = ghostState.getPosition()
        distanceToGhost = manhattanDistance(myPosition, ghostPos)

        if (distanceToGhost == 0):
            score -= 1000
        else:
            # Check scared ghosts. If this ghost is scared AND pacman is within
            # distance to the ghost, then he should try to eat it
            scaredTime = ghostState.scaredTimer
            if (scaredTime > 0 and distanceToGhost < scaredTime):
                scaredGhostScore += (1.0 / distanceToGhost)

            # Otherwise, the ghost is not scared, so pacman should keep away
            else:
                if (distanceToGhost < 3):
                    ghostScore -= (1.0 / distanceToGhost)

    # scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    # scaredGhostScore += sum(scaredTimes)

    # Case 2: proximity to food
    foodScore = 0
    for foodPos in foodLst:
        distanceToFood = manhattanDistance(myPosition, foodPos)
        if (distanceToFood == 0):
            foodScore += 1
        else:
            foodScore += (1.0 / distanceToFood)

    # Case 3: proximity to capsules
    capsuleScore = 0
    for capsulePos in currentGameState.getCapsules():
        distanceToCapsule = manhattanDistance(myPosition, capsulePos)
        if (distanceToCapsule == 0):
            capsuleScore += 1
        else:
            capsuleScore += (1.0 / distanceToCapsule)


    # Make a linear combination with weights of importance
    # NOTE: I did a lot of trial and error here. These weights seem to get the max
    # outcome on the autograder tests
    score += (ghostScore * 4.5 + scaredGhostScore * 9 + foodScore * 2 + capsuleScore * 3)
    return score





# Abbreviation
better = betterEvaluationFunction
