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
from game import Directions
import game
import numpy as np


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='MinimaxAgent', second='MinimaxAgent'):
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
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


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

    def get_agent_distance_features(self, enemy_pos, friend_pos):
        enemy_dists = [[self.dist(f, e) for e in enemy_pos] for f in friend_pos]
        friend_dists = ([([self.dist(f, e) for e in enemy_pos if f != e]) for f in friend_pos])

        closest_enemy = min([min(d) for d in enemy_dists])
        mean_closest_enemy = np.mean([min(d) for d in enemy_dists])
        min_mean_dist_enemy = min(np.mean(d) for d in enemy_dists)

        closest_friend = min([min(d) for d in friend_dists])
        mean_closest_friend = np.mean([min(d) for d in friend_diswts])
        min_mean_dist_friend = min(np.mean(d) for d in enemy_friends)

        return closest_enemy, mean_closest_enemy, min_mean_dist_enemy, closest_friend, mean_closest_friend, min_mean_dist_friend

    def get_friendly_food_features(self, friends_pos, friendly_food):
        food = friendly_food.asList()
        friends = friends_pos

        #print(food)
        num_food = len(food)
        dist_to_min_food = min([min([self.dist(foo, f) for foo in food]) for f in friends])
        farthest_min_dis_to_food = max([min([self.dist(foo, f) for foo in food]) for f in friends])
        mean_min_dist_to_food = np.mean([min([self.dist(foo, f) for foo in food]) for f in friends])
        mean_dist_to_food = np.mean([[self.dist(foo, f) for foo in food] for f in friends])


        return num_food, dist_to_min_food, farthest_min_dis_to_food, mean_min_dist_to_food, mean_dist_to_food

    def get_enemy_food_features(self, friends_pos, enemy_food):
        food = enemy_food.asList()
        friends = friends_pos

        #print(food)
        num_enemy_food = len(food)
        dist_to_min_enemy_food = min([min([self.dist(foo, f) for foo in food]) for f in friends])
        farthest_min_dis_to_enemy_food = max([min([self.dist(foo, f) for foo in food]) for f in friends])
        mean_min_dist_to_enemy_food = np.mean([min([self.dist(foo, f) for foo in food]) for f in friends])
        mean_dist_to_enemy_food = np.mean([[self.dist(foo, f) for foo in food] for f in friends])


        return num_enemy_food, dist_to_min_enemy_food, farthest_min_dis_to_enemy_food, mean_min_dist_to_enemy_food, mean_dist_to_enemy_food

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
        epsilon = 30


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
        enemy_food = self.getFoodYouAreDefending(gameState)
        friendly_food = self.getFood(gameState)


        num_food, dist_to_min_food, farthest_min_dis_to_food, mean_min_dist_to_food, mean_dist_to_food = self.get_friendly_food_features(friend_pos, enemy_food)
        num_enemy_food, dist_to_min_enemy_food, farthest_min_dis_to_enemy_food, mean_min_dist_to_enemy_food, mean_dist_to_enemy_food = self.get_enemy_food_features(friend_pos, friendly_food)
        result = 0.1*score - dist_to_min_food - farthest_min_dis_to_food - mean_dist_to_food - mean_dist_to_food + 5*num_food
        result = result - 5*num_enemy_food - dist_to_min_enemy_food - farthest_min_dis_to_enemy_food - mean_min_dist_to_enemy_food - mean_dist_to_enemy_food
        print(result)
        return result

    def alphaBetaHelper(self, gameState, depth, evalFunc, agent_index, alpha, beta, root_index):

        # if gameState.isWin():
        #    return evalFunc(gameState), None

        # if gameState.isLose():
        #    return evalFunc(gameState), None

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
