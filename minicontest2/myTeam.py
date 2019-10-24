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
        self.recent2 = util.Queue()

    def update_recent(self, pos):
        self.recent.push(pos)
        self.recent2.push(pos)
        if len(self.recent.list) > 15:
            self.recent.pop()
        if len(self.recent2.list) > 125:
            self.recent2.pop()

    def num_in_recent(self, pos):
        return len([p for p in self.recent.list if p == pos]), len([p for p in self.recent2.list if p == pos])

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
            if d >= 4:
                continue
            if self.is_in_enemy(gameState, e):
                r = -1/(d-0.9)
                enemy_dists.append(r)
            else:
                enemy_dists.append(1/(d+1))

        if len(enemy_dists) <= 0:
            return 0
        return sum(enemy_dists)


    def get_enemy_food_features(self, gameState, enemy_food):
        food = enemy_food.asList()

        num_food = len(food)
        agent_pos = gameState.getAgentPosition(self.index)

        mh_graph = [[0 for i in range(len(food))] for i in range(len(food))]
        for i in range(len(food)):
            for j in range(i + 1, len(food)):
                mh_graph[i][j] = self.dist(food[i], food[j])
                mh_graph[j][i] = mh_graph[i][j]

        X = mh_graph # csr_matrix(mh_graph)

        Tcsr = prims(X)

        min_dist_to_food = min([self.dist(agent_pos, f) for f in food])

        return num_food, deep_sum(Tcsr), min_dist_to_food

    def get_friendly_food_features(self, gameState, friend_food):
        food = friend_food.asList()

        num_food = len(food)
        agent_pos = gameState.getAgentPosition(self.index)

        mh_graph = [[0 for i in range(len(food))] for i in range(len(food))]
        for i in range(len(food)):
            for j in range(i + 1, len(food)):
                mh_graph[i][j] = self.dist(food[i], food[j])

        X = mh_graph# csr_matrix(mh_graph)

        Tcsr = prims(X)

        min_dist_to_food = min([self.dist(agent_pos, f) for f in food])

        return num_food, deep_sum(Tcsr), min_dist_to_food

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

        #print(results)
        optomizing_arg = argmax(results)
        return actions[optomizing_arg]

    def evaluate(self, gameState, action):
        weights = self.getWeights(gameState, action)
        features = self.getFeatures(gameState, action)

        tot = 0
        for k in weights.keys():
            tot += weights[k] * features[k]

        if action == "Stop":
            return tot - 10000000
        return tot

    def getWeights(self, gameState, action):
        # Set this manually
        result = dict()

        result["score"] = 10
        result["num_enemy_food"] = -150
        result["enemy_mst_sum"] = -100
        result["min_dist_to_food"] = -10
        result["enemy_dists"] = 85
        result["remaining_uncaptured"] = -999999999
        result["carrying_food"] = -1
        result["min_dist_to_friend"] = 80
        result["times_visited"] = -3
        result["num_capsules"] = -1000
        result["min_dist_capsule"] = 50
        result["times_visited_long"] = -1
        result["num_enemies"] = -10000
        #result["max_dist_to_friend_dot"] = 10



        return result

    # Merges all the features together into one dict
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

        #print(self.remaining_uncaptured_foods, self.carrying)

        # print(self.remaining_uncaptured_foods, self.carrying)

        max_dist_to_friend_dot = 0

        min_dist_to_friend = 0
        if self.is_in_enemy(new_gs, new_gs.getAgentPosition(self.index)) and self.carrying >=1:
            min_dist_to_friend = min([self.dist(new_gs.getAgentPosition(self.index), new_gs.getAgentPosition(f)) for f in friends if f != self.index])
            min_dist_to_friend = 1/min_dist_to_friend

        closest_enemy = min(self.dist(new_gs.getAgentPosition(self.index), e) for e in enemy_pos)
        gon_get_got = closest_enemy < (2 * min_dist_to_food)


        if self.carrying >= 1:
            num_friendly_food, enemy_mst_sum, min_dist_to_food = self.get_friendly_food_features(new_gs, enemy_food)
            enemy_mst_sum = enemy_mst_sum**0.5
            max_dist_to_friend_dot = min([self.dist(f, new_gs.getAgentPosition(self.index)) for f in enemy_food.asList()])

        self.update_recent(gameState.getAgentPosition(self.index))
        times_visited, times_visited_long = self.num_in_recent(new_gs.getAgentPosition(self.index))

        num_capsules = len(self.getCapsules(new_gs))
        min_dist_capsule = 0
        if num_capsules:
            min_dist_capsule = min([self.dist(cap, gameState.getAgentPosition(self.index)) for cap in self.getCapsules(new_gs)])
            if self.carrying:
                min_dist_capsule = 3 * min_dist_capsule
        #print(num_capsules, min_dist_capsule)

        if not self.red:
            opps_in_our_territory = [not new_gs.isRed(i) for i in enemy_pos]
        else:
            opps_in_our_territory = [new_gs.isRed(i) for i in enemy_pos]


        #result["max_dist_to_friend_dot"] = 1/(max_dist_to_friend_dot+.01)
        result["num_enemies"] = len(opps_in_our_territory)
        result["num_capsules"] = num_capsules
        result["min_dist_capsule"] = 1/(min_dist_capsule-0.8)
        result["times_visited"] = times_visited**1.35
        result["times_visited_long"] = times_visited_long**1.075
        result["min_dist_to_friend"] = min_dist_to_friend
        result["min_dist_to_food"] = min_dist_to_food
        result["score"] = score
        result["num_enemy_food"] = num_enemy_food
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
        optomizing_arg = argmax(results)
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
        return {"num_opps_in_territory": -100000, "num_food_in_territory": 20, "is_in_enemy": -10000000,
                "min_dist": -5, "min_dist_opp_in_territory": -690}
                # "dist_between": -100}

    def getFeatures(self, gameState, action):
        # figure out good features here
        new_gamestate = gameState.generateSuccessor(self.index, action)
        current_pos = new_gamestate.getAgentPosition(self.index)
        friends = self.getTeam(new_gamestate)
        opps = self.getOpponents(new_gamestate)
        opp_distances = [new_gamestate.getAgentPosition(i) for i in opps]
        friend_pos = [new_gamestate.getAgentPosition(i) for i in friends]
        enemy_pos = [new_gamestate.getAgentPosition(i) for i in opps]
        opps_in_our_territory = []
        opps_in_our_territory_dist = []
        dist_tool = self.distancer
        if not self.red:
            opps_in_our_territory = [not new_gamestate.isRed(i) for i in enemy_pos]
            opps_in_our_territory_dist = [dist_tool.getDistance(i, current_pos) for i in enemy_pos if
                                          not new_gamestate.isRed(i)]
        else:
            opps_in_our_territory = [new_gamestate.isRed(i) for i in enemy_pos]
            opps_in_our_territory_dist = [dist_tool.getDistance(i, current_pos) for i in enemy_pos if
                                          new_gamestate.isRed(i)]
        dist_between = abs(
            dist_tool.getDistance(enemy_pos[0], current_pos) - dist_tool.getDistance(enemy_pos[1], current_pos))
        sum_opps = 0
        if len(opps_in_our_territory) != 0:
            sum_opps = sum(opps_in_our_territory)
        friendly_food = sum([sum(i) for i in self.getFood(new_gamestate)])
        is_in_opp_ground = 0
        if self.is_in_enemy(new_gamestate, current_pos):
            is_in_opp_ground = 1
        distance_from_opp = [self.distancer.getDistance(i, current_pos) for i in
                             [new_gamestate.getAgentPosition(k) for k in opps]]
        min_dist_from_opp = min(distance_from_opp)
        if min_dist_from_opp == 0:
            min_dist_from_opp = -1000
        min_dist_opp_in_terr = 0

        if (len(opps_in_our_territory_dist) != 0):
            min_dist_opp_in_terr = min(opps_in_our_territory_dist)
        return {"num_opps_in_territory": sum_opps, "num_food_in_territory": friendly_food,
                "is_in_enemy": is_in_opp_ground, "min_dist": min_dist_from_opp,
                "min_dist_opp_in_territory": min_dist_opp_in_terr,"dist_between": dist_between}


def argmax(array):
    #assert len(array) <= 0
    return array.index(max(array))


# From Coderbyte
def prims(adjMatrix):
    V = len(adjMatrix)

    # arbitrarily choose initial vertex from graph
    vertex = 0

    # initialize empty edges array and empty MST
    MST = []
    edges = []
    visited = []
    minEdge = [None, None, float('inf')]

    # run prims algorithm until we create an MST
    # that contains every vertex from the graph
    while len(MST) != V - 1:

        # mark this vertex as visited
        visited.append(vertex)

        # add each edge to list of potential edges
        for r in range(0, V):
            if adjMatrix[vertex][r] != 0:
                edges.append([vertex, r, adjMatrix[vertex][r]])

        # find edge with the smallest weight to a vertex
        # that has not yet been visited
        for e in range(0, len(edges)):
            if edges[e][2] < minEdge[2] and edges[e][1] not in visited:
                minEdge = edges[e]

        # remove min weight edge from list of edges
        edges.remove(minEdge)

        # push min edge to MST
        MST.append(minEdge)

        # start at new vertex and reset min edge
        vertex = minEdge[1]
        minEdge = [None, None, float('inf')]

    return MST


# From StackOverflow
def deep_sum(L):
    total = 0  # don't use `sum` as a variable name
    for i in L:
        if isinstance(i, list):  # checks if `i` is a list
            total += deep_sum(i)
        else:
            total += i
    return total
