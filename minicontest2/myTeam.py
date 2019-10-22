# myTeam.py
# ---------
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


from captureAgents import CaptureAgent
import random, time, util
import numpy as np
from pprint import pprint
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveAgent', second='DefensiveAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    # The following line is an example only; feel free to change it.
    return [ eval(first)(firstIndex) , eval(second)(secondIndex) ]


##########
# Agents #
##########

class DummyAgent(CaptureAgent):
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at baselineTeam.py for more details about how to
    create an agent as this is the bare minimum.
    """

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).

        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)

        IMPORTANT: This method may run for at most 15 seconds.
        """

        '''
        Make sure you do not delete the following line. If you would like to
        use Manhattan distances instead of maze distances in order to save
        on initialization time, please take a look at
        CaptureAgent.registerInitialState in captureAgents.py.
        '''
        CaptureAgent.registerInitialState(self, gameState)

        '''
        Your initialization code goes here, if you need any.
        '''

    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        actions = gameState.getLegalActions(self.index)

        '''
        You should change this in your own agent.
        '''

        return random.choice(actions)


class MinimaxAgent(CaptureAgent):

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).

        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)

        IMPORTANT: This method may run for at most 15 seconds.
        """

        '''
        Make sure you do not delete the following line. If you would like to
        use Manhattan distances instead of maze distances in order to save
        on initialization time, please take a look at
        CaptureAgent.registerInitialState in captureAgents.py.
        '''
        CaptureAgent.registerInitialState(self, gameState)

        '''
        Your initialization code goes here, if you need any.
        '''

    def dist(self, coord1, coord2):
        return self.distancer.getDistance(coord1, coord2)

    def is_in_enemy(self, gameState, pos):
        if self.red:
            return not gameState.isRed(pos)
        return gameState.isRed(pos)

    def get_agent_distance_features(self, gameState, enemy_pos, friend_pos):
        enemy_dists = [[self.dist(f, e) for e in enemy_pos] for f in friend_pos]
        #friend_dists = ([([self.dist(f, e) for e in enemy_pos if f != e]) for f in friend_pos])
        enemy_dists = []
        for f in friend_pos:
            for e in enemy_pos:
                d = self.dist(f, e)
                if not self.is_in_enemy(gameState, f):
                    d = -10*d
        return np.sum(enemy_dists)

    def get_friendly_food_features(self, friends_pos, friendly_food):
        food = friendly_food.asList()

        num_food = len(food)
        # mean_dist_to_food = np.mean([[self.dist(foo, f) for foo in food] for f in friends_pos])
        food = food + friends_pos


        mh_graph = np.zeros((len(food), len(food)))
        for i in range(len(food)):
            for j in range(i + 1, len(food)):
                mh_graph[i, j] = self.dist(food[i], food[j])

        X = csr_matrix(mh_graph)

        Tcsr = minimum_spanning_tree(X)

        return num_food, np.sum(Tcsr)#, mean_dist_to_food


    def get_enemy_food_features(self, friends_pos, enemy_food):
        food = enemy_food.asList()

        num_food = len(food)

        food = food + friends_pos

        mh_graph = np.zeros((len(food), len(food)))
        for i in range(len(food)):
            for j in range(i + 1, len(food)):
                mh_graph[i, j] = self.dist(food[i], food[j])

        X = csr_matrix(mh_graph)

        Tcsr = minimum_spanning_tree(X)

        return num_food, np.sum(Tcsr)



    '''
    def getFeatures(self, gameState):

        ## Features between agents
        enemies = self.getOpponents(gameState)
        friends = self.getTeam(gameState)

        score = self.getScore(gameState)

        friend_pos = [gameState.getAgentPosition(i) for i in friends]
        enemy_pos = [gameState.getAgentPosition(i) for i in enemies]

        closest_enemy, mean_closest_enemy, min_mean_dist_enemy, closest_friend, mean_closest_friend, min_mean_dist_friend = self.get_agent_distance_features(enemy_pos, friend_pos)

        # Features of agents in each others territory

        if self.blue:
            enemies_in_our_territory = [gameState.isBlue(i) for i in enemy_pos]
        else:
            enemies_in_our_territory = [gameState.isRed(i) for i in enemy_pos]

        if self.blue:
            friends_in_enemy_territory = [gameState.isRed(i) for i in friend_pos]
        else:
            friends_in_enemy_territory = [gameState.isBlue(i) for i in friend_pos]

        prop_enemies_in_us = np.mean(enemies_in_our_territory)
        prop_friends_in_them = np.mean(friends_in_enemy_territory)

        num_enemies_in_us = sum(enemies_in_our_territory)
        num_friends_in_them = sum(friends_in_enemy_territory)

        # Features of agents in their territory
        if self.blue:
            friends_in_our_territory = [gameState.isBlue(i) for i in friend_pos]
        else:
            friends_in_out_territory = [gameState.isRed(i) for i in friend_pos]

        if self.blue:
            enemies_in_their_territory = [gameState.isRed(i) for i in friend_pos]
        else:
            enemies_in_their_territory = [gameState.isBlue(i) for i in friend_pos]

        # Features between foods and agents
        enemy_food = CaptureAgent.getFoodYouAreDefending(gameState)
        friendly_food = CaptureAgent.getFood(gameState)

        # Features between Super-Capsules an Agents
        enemy_capsules = CaptureAgent.getCapsules(gameState)
        friendly_capsule = CaptureAgent.getCapsulesYouAreDefending(gameState)

        return score, prop_enemies_in_us, prop_friends_in_them, num_enemies_in_us, num_friends_in_them
    '''


    def is_friend(self, agent_index, gameState):
        return agent_index in self.getTeam(gameState)

    def is_enemy(self, agent_index, gameState):
        return agent_index in self.getOpponents(gameState)

    def are_friends(self, index1, index2, gameState):
        one = index1 in self.getTeam(gameState)
        two = index2 in self.getTeam(gameState)
        return one == two

    def are_enemies(self, index1, index2, gameState):
        return not self.are_friends(index1, index2, gameState)

    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        actions = gameState.getLegalActions(self.index)
        actions = [a for a in actions if a != "Stop"]

        '''
        You should change this in your own agent.
        '''

        if self.red:
            print("red", self.index, actions)
        else:
            print("blue", self.index, actions)

        # Hyperparams
        depth = 1
        eval_func = self.manual_eval_func
        epsilon = 0


        rand_num = random.randint(0, 100)
        if rand_num < epsilon:
            return np.random.choice(actions)

        action = self.alphaBetaHelper(gameState, depth, eval_func, self.index, float("-inf"), float("inf"), self.index)[
            1]
        return action

    def manual_eval_func(self, gameState):
        score = self.getScore(gameState)
        enemies = self.getOpponents(gameState)
        friends = self.getTeam(gameState)
        friend_pos = [gameState.getAgentPosition(i) for i in friends]
        enemy_pos = [gameState.getAgentPosition(i) for i in enemies]
        enemy_food = self.getFoodYouAreDefending(gameState)
        friendly_food = self.getFood(gameState)


        num_food, friendly_mst_sum= self.get_friendly_food_features(friend_pos, enemy_food)
        num_enemy_food, enemy_mst_sum = self.get_enemy_food_features(friend_pos, friendly_food)
        enemy_dists = self.get_agent_distance_features(gameState, enemy_pos, friend_pos)


        result = 2*score + 10*num_food + 2*friendly_mst_sum
        result = result - 12*num_enemy_food - 2*enemy_mst_sum
        result += enemy_dists

        print(result)

        return result


    def alphaBetaHelper(self, gameState, depth, evalFunc, agent_index, alpha, beta, root_index):

        if gameState.isOver():
           return evalFunc(gameState), None

        if depth <= 0:
            return evalFunc(gameState), None

        actions = gameState.getLegalActions(agent_index)
        # actions = [a for a in actions if a != 'Stop']

        new_depth = depth
        num_agents = gameState.getNumAgents()
        new_index = (agent_index + 1) % num_agents

        if new_index == root_index - 1:
            new_depth = depth - 1

        succesors_scores = []

        for action in actions:
            succesor = gameState.generateSuccessor(agent_index, action)

            val = self.alphaBetaHelper(succesor, new_depth, evalFunc, new_index, alpha, beta, root_index)[0]
            if self.is_friend(agent_index, gameState):
                if val > beta:
                    return val, action
                alpha = max(alpha, val)
            else:
                if val < alpha:
                    return val, action
                beta = min(beta, val)

            succesors_scores.append((val, action))

        succesors_acts = np.array([score[1] for score in succesors_scores])
        succesors_scores = np.array([score[0] for score in succesors_scores])

        if self.is_friend(agent_index, gameState):
            optimizing_arg = np.argmax(succesors_scores)
        else:
            optimizing_arg = np.argmin(succesors_scores)
        return succesors_scores[optimizing_arg], succesors_acts[optimizing_arg]



class OffensiveAgent(CaptureAgent):
    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).

        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)

        IMPORTANT: This method may run for at most 15 seconds.
        """

        '''
        Make sure you do not delete the following line. If you would like to
        use Manhattan distances instead of maze distances in order to save
        on initialization time, please take a look at
        CaptureAgent.registerInitialState in captureAgents.py.
        '''
        CaptureAgent.registerInitialState(self, gameState)

        '''
        Your initialization code goes here, if you need any.
        '''
        self.remaining_uncaptured_foods = 0
        self.carrying = 0

    def is_friend(self, agent_index, gameState):
        return agent_index in self.getTeam(gameState)

    def is_enemy(self, agent_index, gameState):
        return agent_index in self.getOpponents(gameState)

    def are_friends(self, index1, index2, gameState):
        one = index1 in self.getTeam(gameState)
        two = index2 in self.getTeam(gameState)
        return one == two

    def are_enemies(self, index1, index2, gameState):
        return not self.are_friends(index1, index2, gameState)


    def get_agent_distance_features(self, gameState, enemy_pos, friend_pos):
        #enemy_dists = [[self.dist(f, e) for e in enemy_pos] for f in friend_pos]
        #friend_dists = ([([self.dist(f, e) for e in enemy_pos if f != e]) for f in friend_pos])

        enemy_dists = []
        for f in friend_pos:
            for e in enemy_pos:
                d = self.dist(f, e)
                if d > 7:
                    continue
                if not self.is_in_enemy(gameState, f):
                    d = -10*d
                else:
                    d = 0
                enemy_dists.append(d)


        if len(enemy_dists) <= 0:
            return 0
        return np.sum(enemy_dists)


    def get_enemy_food_features(self, gameState, enemy_food):
        food = enemy_food.asList()

        num_food = len(food)
        agent_pos = gameState.getAgentPosition(self.index)

        mh_graph = np.zeros((len(food), len(food)))
        for i in range(len(food)):
            for j in range(i + 1, len(food)):
                mh_graph[i, j] = self.dist(food[i], food[j])

        X = csr_matrix(mh_graph)

        Tcsr = minimum_spanning_tree(X)

        min_dist_to_food = min([self.dist(agent_pos, f) for f in food])

        return num_food, np.sum(Tcsr), min_dist_to_food

    def get_friendly_food_features(self, gameState, friend_food):
        food = friend_food.asList()

        num_food = len(food)
        agent_pos = gameState.getAgentPosition(self.index)

        mh_graph = np.zeros((len(food), len(food)))
        for i in range(len(food)):
            for j in range(i + 1, len(food)):
                mh_graph[i, j] = self.dist(food[i], food[j])

        X = csr_matrix(mh_graph)

        Tcsr = minimum_spanning_tree(X)

        min_dist_to_food = min([self.dist(agent_pos, f) for f in food])

        return num_food, np.sum(Tcsr), min_dist_to_food

    def dist(self, coord1, coord2):
        return self.distancer.getDistance(coord1, coord2)

    def is_in_enemy(self, gameState, pos):
        if self.red:
            return not gameState.isRed(pos)
        return gameState.isRed(pos)



    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        actions = gameState.getLegalActions(self.index)
        #actions = [a for a in actions if a != "Stop"]

        '''
        You should change this in your own agent.
        '''
        results = []
        for action in actions:
            results.append(self.evaluate(gameState, action))

        optomizing_arg = np.argmax(results)
        return actions[optomizing_arg]

    def evaluate(self, gameState, action):
        weights = self.getWeights(gameState, action)
        features = self.getFeatures(gameState, action)

        sum = 0
        for k in weights.keys():
            sum += weights[k] * features[k]

        return sum

    def getWeights(self, gameState, action):
        # Set this manually
        result = dict()

        result["score"] = 0.1
        result["num_enemy_food"] = -1000
        result["enemy_mst_sum"] = -100
        result["min_dist_to_food"] = -10
        result["enemy_dists"] = 0
        result["remaining_uncaptured"] = -100000
        result["carrying_food"] = 0
        #result["max_dist_to_friend_dot"] = 10



        return result

    def getFeatures(self, gameState, action):
        result = dict()
        new_gs = gameState.generateSuccessor(self.index, action)

        # figure out good features here
        score = self.getScore(new_gs)
        enemies = self.getOpponents(new_gs)
        friends = self.getTeam(new_gs)
        friend_pos = [new_gs.getAgentPosition(i) for i in friends]
        curr_friend_pos = [gameState.getAgentPosition(i) for i in friends]
        enemy_pos = [new_gs.getAgentPosition(i) for i in enemies]
        friendly_food = self.getFood(new_gs)
        enemy_food = self.getFoodYouAreDefending(new_gs)

        num_enemy_food, enemy_mst_sum, min_dist_to_food = self.get_enemy_food_features(new_gs, friendly_food)
        enemy_dists = self.get_agent_distance_features(new_gs, enemy_pos, friend_pos)

        if sum([self.is_in_enemy(gameState, f) for f in curr_friend_pos]) <= 0:
            self.remaining_uncaptured_foods = num_enemy_food

        self.carrying = self.remaining_uncaptured_foods - num_enemy_food

        print(self.remaining_uncaptured_foods, self.carrying)

        #print(self.remaining_uncaptured_foods, self.carrying)

        max_dist_to_friend_dot = 0


        if self.carrying >= 1:
            num_friendly_food, enemy_mst_sum, min_dist_to_food = self.get_friendly_food_features(new_gs, enemy_food)
            max_dist_to_friend_dot = min([self.dist(f, new_gs.getAgentPosition(self.index)) for f in enemy_food.asList()])


        #result["max_dist_to_friend_dot"] = 1/(max_dist_to_friend_dot+.01)
        result["min_dist_to_food"] = min_dist_to_food
        result["score"] = score
        result["num_enemy_food"] = num_enemy_food
        result["min_dist_to_food"] = min_dist_to_food
        result["enemy_mst_sum"] = enemy_mst_sum
        result["enemy_dists"] = enemy_dists
        result["remaining_uncaptured"] = self.remaining_uncaptured_foods
        result["carrying_food"] = self.carrying


        return result



class DefensiveAgent(CaptureAgent):

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).

        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)

        IMPORTANT: This method may run for at most 15 seconds.
        """

        '''
        Make sure you do not delete the following line. If you would like to
        use Manhattan distances instead of maze distances in order to save
        on initialization time, please take a look at
        CaptureAgent.registerInitialState in captureAgents.py.
        '''
        CaptureAgent.registerInitialState(self, gameState)
        '''
        Your initialization code goes here, if you need any.
        '''
    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        actions = gameState.getLegalActions(self.index)
        actions = [a for a in actions if a != "Stop"]

        '''
        You should change this in your own agent.
        '''
        results = []
        for action in actions:
            results.append(self.evaluate(gameState, action))
        optomizing_arg = np.argmax(results)
        return actions[optomizing_arg]

    def evaluate(self, gameState, action):
        weights = self.getWeights(gameState, action)
        features = self.getFeatures(gameState, action)

        sum = 0
        for k in weights.keys():
            sum += weights[k] * features[k]

        return sum
    def is_in_enemy(self, gameState, pos):
        if self.red:
            return not gameState.isRed(pos)
        return gameState.isRed(pos)
    def getWeights(self, gameState, action):
        # Set this manually
        return {"num_opps_in_territory":-10,"num_food_in_territory":15,"is_in_enemy":-1000000000000,"min_dist":-5}

    def getFeatures(self, gameState, action):
        # figure out good features here
        new_gamestate=gameState.generateSuccessor(self.index,action)
        friends = self.getTeam(new_gamestate)
        opp_distances=[new_gamestate.getAgentPosition(i) for i in self.getOpponents(new_gamestate)]
        friend_pos = [new_gamestate.getAgentPosition(i) for i in friends]
        friends_in_our_territory=[]
        if not self.red:
            friends_in_our_territory = [ not new_gamestate.isRed(i) for i in friend_pos]
        else:
            friends_in_out_territory = [new_gamestate.isRed(i) for i in friend_pos]
        sum_opps=0
        if len(friends_in_our_territory)!=0:
            sum_opps=sum(friends_in_our_territory)
        friendly_food = sum([sum(i) for i in self.getFood(new_gamestate)])
        is_in_opp_ground=0
        if self.is_in_enemy(new_gamestate, new_gamestate.getAgentPosition(self.index)):
            is_in_opp_ground=1
        x=[self.distancer.getDistance(i,new_gamestate.getAgentPosition(self.index)) for i in [new_gamestate.getAgentPosition(k) for k in self.getOpponents(new_gamestate)]]
        min_dist_from_opp=min(x)
        if min_dist_from_opp==0:
            min_dist_from_opp= -500
        return {"num_opps_in_territory":sum_opps,"num_food_in_territory":friendly_food,"is_in_enemy":is_in_opp_ground,"min_dist":min_dist_from_opp}
