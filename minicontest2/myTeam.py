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
        self.recent = util.Queue()

    def update_recent(self, pos):
        self.recent.push(pos)
        if len(self.recent.list) > 30:
            self.recent.pop()

    def num_in_recent(self, pos):
        return len([p for p in self.recent.list if p == pos])

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
        f = gameState.getAgentPosition(self.index)
        for e in enemy_pos:
            d = self.dist(f, e)
            if d > 4:
                continue
            if self.is_in_enemy(gameState, e):
                r = -1/(d-0.5)
                enemy_dists.append(r)
            else:
                enemy_dists.append(1/(d+1))

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

        if action == "Stop":
            return sum - 10000000
        return sum

    def getWeights(self, gameState, action):
        # Set this manually
        result = dict()

        result["score"] = 10
        result["num_enemy_food"] = -75000
        result["enemy_mst_sum"] = -500
        result["min_dist_to_food"] = -10
        result["enemy_dists"] = 50
        result["remaining_uncaptured"] = -999999999
        result["carrying_food"] = -1
        result["min_dist_to_friend"] = 100
        result["times_visited"] = -10000
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

        min_dist_to_friend = 0
        if self.is_in_enemy(new_gs, new_gs.getAgentPosition(self.index)) and self.carrying >=1:
            min_dist_to_friend = min([self.dist(new_gs.getAgentPosition(self.index), new_gs.getAgentPosition(f)) for f in friends if f != self.index])
            min_dist_to_friend = 1/ min_dist_to_friend


        if self.carrying >= 1:
            num_friendly_food, enemy_mst_sum, min_dist_to_food = self.get_friendly_food_features(new_gs, enemy_food)
            enemy_mst_sum = enemy_mst_sum**0.5
            max_dist_to_friend_dot = min([self.dist(f, new_gs.getAgentPosition(self.index)) for f in enemy_food.asList()])


        self.update_recent(gameState.getAgentPosition(self.index))
        times_visited = self.num_in_recent(new_gs.getAgentPosition(self.index))


        #result["max_dist_to_friend_dot"] = 1/(max_dist_to_friend_dot+.01)
        result["times_visited"] = 2**times_visited
        result["min_dist_to_friend"] = min_dist_to_friend
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
        return {"num_opps_in_territory":-10000000,"num_food_in_territory":20,"is_in_enemy":-1000000000,"min_dist":-5,"min_dist_opp_in_territory":-400}

    def getFeatures(self, gameState, action):
        # figure out good features here
        new_gamestate=gameState.generateSuccessor(self.index,action)
        friends = self.getTeam(new_gamestate)
        opp_distances=[new_gamestate.getAgentPosition(i) for i in self.getOpponents(new_gamestate)]
        friend_pos = [new_gamestate.getAgentPosition(i) for i in friends]
        enemy_pos=[new_gamestate.getAgentPosition(i) for i in self.getOpponents(new_gamestate)]
        opps_in_our_territory=[]
        opps_in_our_territory_dist=[]
        if not self.red:
            opps_in_our_territory = [ not new_gamestate.isRed(i) for i in enemy_pos]
            opps_in_our_territory_dist = [self.distancer.getDistance(i,new_gamestate.getAgentPosition(self.index)) for i in enemy_pos if not new_gamestate.isRed(i)]
        else:
            opps_in_out_territory = [new_gamestate.isRed(i) for i in enemy_pos]
            opps_in_our_territory_dist = [self.distancer.getDistance(i,new_gamestate.getAgentPosition(self.index)) for i in enemy_pos if new_gamestate.isRed(i)]

        sum_opps=0
        if len(opps_in_our_territory)!=0:
            sum_opps=sum(opps_in_our_territory)
        friendly_food = sum([sum(i) for i in self.getFood(new_gamestate)])
        is_in_opp_ground=0
        if self.is_in_enemy(new_gamestate, new_gamestate.getAgentPosition(self.index)):
            is_in_opp_ground=1
        x=[self.distancer.getDistance(i,new_gamestate.getAgentPosition(self.index)) for i in [new_gamestate.getAgentPosition(k) for k in self.getOpponents(new_gamestate)]]
        min_dist_from_opp=min(x)
        if min_dist_from_opp==0:
            min_dist_from_opp= -1000
        min_dist_opp_in_terr=0
        if(len(opps_in_our_territory_dist)!=0):
            min_dist_opp_in_terr=min(opps_in_our_territory_dist)
        return {"num_opps_in_territory":sum_opps,"num_food_in_territory":friendly_food,"is_in_enemy":is_in_opp_ground,"min_dist":min_dist_from_opp,"min_dist_opp_in_territory":min_dist_opp_in_terr}
