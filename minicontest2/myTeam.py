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
               first='QLearningAgent', second='QLearningAgent'):
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
        mean_closest_friend = np.mean([min(d) for d in friend_dists])
        min_mean_dist_friend = min(np.mean(d) for d in enemy_friends)

        return closest_enemy, mean_closest_enemy, min_mean_dist_enemy, closest_friend, mean_closest_friend, min_mean_dist_friend

    def getFeatures(self, gameState):

        ## Features between agents
        enemies = self.getOppenents(gameState)
        friends = self.getTeam(gameState)

        score = self.getScore(gameState)

        friend_pos = [gameState.getAgentPosition(i) for i in friends]
        enemy_pos = [gameState.getAgentPosition(i) for i in enemies]

        closest_enemy, mean_closest_enemy, min_mean_dist_enemy, closest_friend, mean_closest_friend, min_mean_dist_friend = self.get_agent_distance_features(
            enemy_pos, friend_pos)

        ## Features of agents in enemy territory

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

        ##Features of agents in Friendly Territory

        ##Features between foods and agents
        enemy_food = CaptureAgent.getFoodYouAreDefending(gameState)
        friendly_food = CaptureAgent.getFood(gameState)

        ##Features between Super-Capsules an Agents
        enemy_capsules = CaptureAgent.getCapsules(gameState)
        friendly_capsule = CaptureAgent.getCapsulesYouAreDefending(gameState)

        return score, prop_enemies_in_us, prop_friends_in_them, num_enemies_in_us, num_friends_in_them

    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        actions = gameState.getLegalActions(self.index)

        '''
        You should change this in your own agent.
        '''

        print(actions)

        return random.choice(actions)
