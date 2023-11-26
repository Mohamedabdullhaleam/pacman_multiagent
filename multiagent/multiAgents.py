# multiAgents.py
# --------------


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        ##---------------------------------------------------------------##
        # focusing on eating food.When ghost near don't go,
        # Get remaining food and check if it's eaten
        # Extract information about remaining food
        remainingFoodList = newFood.asList()
        remainingFoodCount = len(remainingFoodList)

        # Calculate the minimum distance to the nearest food pellet
        closestFoodDistance = float("inf")
        for foodPosition in remainingFoodList:
            closestFoodDistance = min(closestFoodDistance, manhattanDistance(newPos, foodPosition))

        # Consider distances to scared ghosts
        ghostDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        closestGhostDistance = min(ghostDistances)

        # Maintain a balance between seeking food and avoiding ghosts
        if closestGhostDistance < 2 and newScaredTimes[0] == 0:
            return -float('inf')  # Penalize if too close to non-scared ghosts

        # Evaluate based on the reciprocal of the distance to the nearest food pellet
        # return successorGameState.getScore() + 1.0 / closestFoodDistance - 10.0 * remainingFoodCount + 100.0 / (closestGhostDistance + 1)
        return successorGameState.getScore() + 1.0 / closestFoodDistance

        ##--------------------------------------------------------------##

def scoreEvaluationFunction(currentGameState: GameState):
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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.maxvalue(gameState, 0, 0)[0]

    def minimax(self, gameState, agentIndex, depth):
        # Check if the depth limit or terminal state is reached
        if depth == self.depth * gameState.getNumAgents() or gameState.isLose() or gameState.isWin():
            # Return the evaluation value if the game state is terminal
            return self.evaluationFunction(gameState)

        # If it's Pacman's turn (maximizing agent)
        if agentIndex == 0:
            return self.maxvalue(gameState, agentIndex, depth)[1]
        # If it's a ghost's turn (minimizing agent)
        else:
            return self.minvalue(gameState, agentIndex, depth)[1]

    def maxvalue(self, gameState, agentIndex, depth):
        bestAction = ("max", float("-inf"))

        for action in gameState.getLegalActions(agentIndex):
            # Generate successor state based on the current action
            successor = gameState.generateSuccessor(agentIndex, action)

            # Evaluate the successor using minimax
            succValue = self.minimax(successor, (depth + 1) % gameState.getNumAgents(), depth + 1)

            succAction = (action, succValue)
            # Update the best action using the maximum value
            bestAction = max(bestAction, succAction, key=lambda x: x[1])

        return bestAction

    def minvalue(self, gameState, agentIndex, depth):
        bestAction = ("min", float("inf"))

        for action in gameState.getLegalActions(agentIndex):
            # Generate successor state based on the current action
            successor = gameState.generateSuccessor(agentIndex, action)

            # Evaluate the successor using minimax
            succValue = self.minimax(successor, (depth + 1) % gameState.getNumAgents(), depth + 1)

            succAction = (action, succValue)
            # Update the best action using the minimum value
            bestAction = min(bestAction, succAction, key=lambda x: x[1])

        return bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.maxval(gameState, 0, 0, -float("inf"), float("inf"))[0]

    def alphabeta(self, gameState, agentIndex, depth, alpha, beta):
        # Check if the depth limit or terminal state is reached
        if depth == self.depth * gameState.getNumAgents() or gameState.isLose() or gameState.isWin():
            # Return the evaluation value if the game state is terminal
            return self.evaluationFunction(gameState)

        # If it's Pacman's turn (maximizing agent)
        if agentIndex == 0:
            # Call the maxval function to find the best move and its value
            _,bestValue = self.maxval(gameState, agentIndex, depth, alpha, beta)
            return bestValue
        # If it's a ghost's turn (minimizing agent)
        else:
            # Call the minval function to find the best move and its value
            _,bestValue = self.minval(gameState, agentIndex, depth, alpha, beta)
            return bestValue

    def maxval(self, gameState, agentIndex, depth, alpha, beta):
        # Initialize the best action with a low value
        best_action = ("max", -float("inf"))
        for action in gameState.getLegalActions(agentIndex):
            # Generate successor state based on the current action
            successor = gameState.generateSuccessor(agentIndex, action)
            # Evaluate the successor using alpha-beta pruning
            succ_value = self.alphabeta(successor, (depth + 1) % gameState.getNumAgents(), depth + 1, alpha, beta)
            succ_action = (action, succ_value)
            # Update the best action using the maximum value
            best_action = max(best_action, succ_action, key=lambda x: x[1])
            # Alpha-beta pruning
            if best_action[1] > beta:
                return best_action
            # Update alpha with the maximum value encountered so far
            alpha = max(alpha, best_action[1])
        return best_action

    def minval(self, gameState, agentIndex, depth, alpha, beta):
        # Initialize the best action with a high value
        bestAction = ("min", float("inf"))
        # Iterate over legal actions
        for action in gameState.getLegalActions(agentIndex):
            # Generate successor state
            successor = gameState.generateSuccessor(agentIndex, action)
            # Evaluate the successor using alpha-beta pruning
            succAction = (
            action, self.alphabeta(successor, (depth + 1) % gameState.getNumAgents(), depth + 1, alpha, beta))
            # Update the best action using the minimum value
            bestAction = min(bestAction, succAction, key=lambda x: x[1])
            # Pruning
            if bestAction[1] < alpha:
                return bestAction
            else:
                beta = min(beta, bestAction[1])
        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        ##--------------------------------------------------------------------------------##
        maxDepth = self.depth * gameState.getNumAgents()
        return self.expectimax(gameState, "expect", maxDepth, 0)[0]

    def expectimax(self, gameState, action, depth, agentIndex):

        if depth == 0 or gameState.isLose() or gameState.isWin():
            return (action, self.evaluationFunction(gameState))

        # If it's the turn for Pacman (max agent), return the result of the maxvalue function
        if agentIndex == 0:
            return self.maxvalue(gameState, action, depth, agentIndex)
        # If it's the turn for the ghost (expected value agent), return the result of the expvalue function
        else:
            return self.expvalue(gameState, action, depth, agentIndex)

    def maxvalue(self, gameState, action, depth, agentIndex):
        # Initialize the best action-value pair with a low value
        bestAction = ("max", -float('inf'))
        for legalAction in gameState.getLegalActions(agentIndex):
            # Determine the index of the next agent in the game
            nextAgent = (agentIndex + 1) % gameState.getNumAgents()
            # Decide on the successor action based on the depth
            succAction = action if depth != self.depth * gameState.getNumAgents() else legalAction
            # Evaluate the successor using the expectimax function
            succValue = self.expectimax(gameState.generateSuccessor(agentIndex, legalAction),
                                        succAction, depth - 1, nextAgent)
            # Update the best action-value pair
            bestAction = max(bestAction, (succAction, succValue[1]), key=lambda x: x[1])

        return bestAction

    def expvalue(self, gameState, action, depth, agentIndex):
        legalActions = gameState.getLegalActions(agentIndex)
        probability = 1.0 / len(legalActions)

        averageScore = sum(
            self.expectimax(gameState.generateSuccessor(agentIndex, legalAction), action, depth - 1,
                            (agentIndex + 1) % gameState.getNumAgents())[1]
            for legalAction in legalActions
        ) * probability

        return (action, averageScore)
    ##--------------------------------------------------------------------------------##


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    ##--------------------------------------------------------------------------------##
    pacmanPos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostPositions = currentGameState.getGhostPositions()

    # Calculate the minimum distance to remaining food or set to a default value if no food remains
    minFoodDist = min(manhattanDistance(pacmanPos, food) for food in foodList) if foodList else 0

    # Calculate the minimum distance to ghosts
    minGhostDist = min(manhattanDistance(pacmanPos, ghost) for ghost in ghostPositions) if ghostPositions else 0

    # Check if any ghost is too close
    if minGhostDist < 2:
        return -float('inf')

    # Get remaining food and capsules counts
    remainingFood = currentGameState.getNumFood()
    remainingCapsules = len(currentGameState.getCapsules())

    # Adjusted weights based on importance
    foodWeight = 950050
    capsulesWeight = 10000
    foodDistWeight = 950

    # Additional factors for win/lose conditions
    additionalFactors = 0
    if currentGameState.isLose():
        additionalFactors -= 50000
    elif currentGameState.isWin():
        additionalFactors += 50000

    # Dynamic multiplier based on the number of remaining food pellets
    dynamicMultiplier = min(1.0, remainingFood / 10.0)

    # Combine the weighted factors to form the evaluation
    return (
            1.0 / (remainingFood + 1) * foodWeight +
            minGhostDist +
            dynamicMultiplier * (1.0 / (minFoodDist + 1) * foodDistWeight) +
            1.0 / (remainingCapsules + 1) * capsulesWeight +
            additionalFactors
    )
# Abbreviation
better = betterEvaluationFunction
    ##--------------------------------------------------------------------------------##


