############################################################
# CSE 190 Final
# Name : Albert Richie
# Email: arichie@ucsd.edu
# PID  : A98407956
# Filename: qlearning.py
############################################################

import rospy
from read_config import read_config
from cse_190_assi_final.msg import *
from copy import deepcopy
import random
import operator
import math
from bisect import bisect
from decimal import *
import optparse

DEBUG_MODE = False

class qlearning():
    
    def __init__(self):

        self.config = read_config()

        # Topics published
        self.policy_list_publish = rospy.Publisher(
            "/results/policy_list",
            PolicyList,
            queue_size=10)

        # Value for each map position
        self.map_dict = {
            "MAP_START": 0,
            "MAP_GOAL": 1,
            "MAP_WALL": 2,
            "MAP_PIT": 3,
            "MAP_PATH": 4 
            }

        self.qvalue_dict = {
            "QVALUE_EAST": 0,
            "QVALUE_WEST": 1,
            "QVALUE_NORTH": 2,
            "QVALUE_SOUTH": 3,
            }

        # Value for each policy map position
        self.policy_dict = {
            "POLICY_EAST": "E",
            "POLICY_WEST": "W",
            "POLICY_NORTH": "N",
            "POLICY_SOUTH": "S",
            "POLICY_WALL": "WALL",
            "POLICY_GOAL": "GOAL",
            "POLICY_PIT": "PIT"
            }

        self.load_configuration()
        self.populate_map()
        if (DEBUG_MODE):
            print self.map
        self.initiate()

############################################################
    # Class functions to publish message outbound
    def publish_policy_list(self):

        publish_policy = PolicyList()
        publish_policy_list = [item for sublist in self.policy_map for item in sublist]
        publish_policy.data = publish_policy_list
        rospy.sleep(2)
        self.policy_list_publish.publish(publish_policy)
        
############################################################
    # Load all necessary parameters from configuration.json
    def load_configuration(self):

        self.move_list = self.config["move_list"]
        self.map_size = self.config["map_size"]
        self.start = self.config["start"]
        self.goal = self.config["goal"]
        self.walls = self.config["walls"]
        self.pits = self.config["pits"]
        self.max_iterations = self.config["max_iterations"]
        self.threshold_difference = self.config["threshold_difference"]
        self.reward_for_each_step = self.config["reward_for_each_step"]
        self.reward_for_hitting_wall = self.config["reward_for_hitting_wall"]
        self.reward_for_reaching_goal = self.config["reward_for_reaching_goal"]
        self.reward_for_falling_in_pit = self.config["reward_for_falling_in_pit"]
        self.discount_factor = self.config["discount_factor"]
        self.prob_move_forward = self.config["prob_move_forward"]
        self.prob_move_backward = self.config["prob_move_backward"]
        self.prob_move_left = self.config["prob_move_left"]
        self.prob_move_right = self.config["prob_move_right"]
        self.position = self.config["start"]
        #self.reward_for_hitting_wall = -0.1
        #self.reward_for_each_step = 0.0
        self.epsilon = 0.5
        self.min_epsilon = 0.1
        self.decay = 1.0
        self.alpha = 0.2
        self.discount_factor = 0.8
        self.qvalue_iteration = 1000
        self.temperature = 1
        self.min_temperature = 0
        self.current_temperature = 1

############################################################
    # Populate map position with value given and known
    # position of start, goal, wall, and pit. Initialize the
    # value of reward for reaching the goal, hitting the wall,
    # and falling in pit.
    def populate_map(self):
   
        self.map = [[self.map_dict["MAP_PATH"] for i in range(self.map_size[1])] for j in range(self.map_size[0])]
        self.map[self.start[0]][self.start[1]] = self.map_dict["MAP_START"]
        self.map[self.goal[0]][self.goal[1]] = self.map_dict["MAP_GOAL"]
        self.value_map = deepcopy(self.map)
        self.policy_map = [["" for i in range(self.map_size[1])] for j in range(self.map_size[0])]
        self.value_map[self.goal[0]][self.goal[1]] = float(self.reward_for_reaching_goal)
        self.visit_map = [[0 for i in range(self.map_size[1])] for j in range(self.map_size[0])]
        self.qvisit_map = [[[1 for i in range(4)] for i in range(self.map_size[1])] for j in range(self.map_size[0])]
        self.qvalue_map = [[[float(0) for i in range(4)] for i in range(self.map_size[1])] for j in range(self.map_size[0])]

        for i, j in self.walls:
            self.map[i][j] = self.map_dict["MAP_WALL"]
            self.value_map[i][j] = float(self.reward_for_hitting_wall)

        for i, j in self.pits:
            self.map[i][j] = self.map_dict["MAP_PIT"]
            self.value_map[i][j] = float(self.reward_for_falling_in_pit)

############################################################
    # Initiate MDP algorithm by calculating the optimal
    # policy for each position. stop value iteration after
    # MDP algorithm converges or it has reached a maximum
    # limit on the number of iterations.
    def initiate(self):
        n = 1
        
        while(n <= self.qvalue_iteration):
            print "Iteration #: %s"%(n)
            #self.epsilon = (1.0)/n
            n = n + 1
            
            self.position = self.start
            self.temp_qvalue_map = deepcopy(self.qvalue_map)
            
            self.decay = self.decay * self.decay
            self.temperature = self.temperature - 0.001

            while (self.position != self.goal and self.position not in self.pits):
                self.handle_move()
                self.visit_map[self.position[0]][self.position[1]] = self.visit_map[self.position[0]][self.position[1]] + 1


            for i, row in enumerate(self.value_map):
                for j, position in enumerate(row):
                    if self.map[i][j] == self.map_dict["MAP_GOAL"]:
                        self.policy_map[i][j] = self.policy_dict["POLICY_GOAL"]
                    elif self.map[i][j] == self.map_dict["MAP_WALL"]:
                        self.policy_map[i][j] = self.policy_dict["POLICY_WALL"]
                    elif self.map[i][j] == self.map_dict["MAP_PIT"]:
                        self.policy_map[i][j] = self.policy_dict["POLICY_PIT"]
                    else:
                        q_index, value = max(enumerate(self.qvalue_map[i][j]), key=operator.itemgetter(1))
                        self.policy_map[i][j] = self.qvalue_policy(q_index)
                        

                    
            qvalue_map = deepcopy(self.temp_qvalue_map)
            self.publish_policy_list()
            
            if n == self.qvalue_iteration :
                #self.publish_policy_list()
                #print self.qvalue_map[2][0]
                break
            
############################################################
    # Handle robot movement until it found the goal or fall
    # into a pit.
    def handle_move(self):

        action = self.get_action_soft_max()
        self.qvisit_map[self.position[0]][self.position[1]][self.move_list_qvalue(action)] = self.qvisit_map[self.position[0]][self.position[1]][self.move_list_qvalue(action)] + 1
        
        next_position = [0, 0]
        
        next_position[0] = self.position[0] + action[0]
        next_position[1] = self.position[1] + action[1]

        if (next_position[0] >= 0 and next_position[0] < len(self.map) and
            next_position[1] >= 0 and next_position[1] < len(self.map[0])):

            if (self.map[next_position[0]][next_position[1]] == self.map_dict["MAP_WALL"]):

                next_position = self.position
                self.qvalue_map[self.position[0]][self.position[1]][self.move_list_qvalue(action)] = max (self.qvalue_map[self.position[0]][self.position[1]]) + \
                                                                                                self.alpha*(self.get_reward(self.position[0], self.position[1], next_position[0], next_position[1])
                                                                                                              - max(self.qvalue_map[self.position[0]][self.position[1]]))
            else:
                self.qvalue_map[self.position[0]][self.position[1]][self.move_list_qvalue(action)] = max (self.qvalue_map[self.position[0]][self.position[1]]) + \
                                                                                                self.alpha*(self.get_reward(self.position[0], self.position[1], next_position[0], next_position[1]) + \
                                                                            self.discount_factor * max(self.qvalue_map[next_position[0]][next_position[1]])- max(self.qvalue_map[self.position[0]][self.position[1]]))
                self.position = next_position
                
            
        else:
            self.handle_move()

############################################################
    # Various exploration vs exploitation function used
    # in q learning algorithm.    
    def get_action_epsilon_decay(self):

        if self.explore(max(self.min_epsilon, self.decay*self.epsilon)):
            return random.choice(self.move_list)
        else:
            index, value = max(enumerate(self.qvalue_map[self.position[0]][self.position[1]]), key=operator.itemgetter(1))
            return self.qvalue_move_list(index)

    def get_action_soft_max(self):

        self.current_temperature = max(self.temperature, self.min_temperature)
        
        if self.current_temperature == 0:
            index, value = max(enumerate(self.qvalue_map[self.position[0]][self.position[1]]), key=operator.itemgetter(1))
            return self.qvalue_move_list(index)
        else:
            choices = []
            total_qvalue = sum([math.e**(value/self.current_temperature) for value in self.qvalue_map[self.position[0]][self.position[1]]])
            for index, qvalue in enumerate(self.qvalue_map[self.position[0]][self.position[1]]):
                probabilities = math.e**(qvalue/self.current_temperature)/total_qvalue
                choices.append([self.qvalue_move_list(index), probabilities])

            return self.weighted_choice(choices)

    def get_action3(self):

        if self.temperature == 0:
            index, value = max(enumerate(self.qvalue_map[self.position[0]][self.position[1]]), key=operator.itemgetter(1))
            return self.qvalue_move_list(index)
        else:
            choices = []
            total_qvalue = sum([math.e**((1/value)/self.temperature) for value in self.qvisit_map[self.position[0]][self.position[1]]])
            for index, qvalue in enumerate(self.qvisit_map[self.position[0]][self.position[1]]):
                probabilities = math.e**((1/qvalue)/self.temperature)/total_qvalue
                choices.append([self.qvalue_move_list(index), probabilities])

            return self.weighted_choice(choices)

############################################################
    # Weighted choice function for softmax function.
    # Will return an action with probability given in an
    # array.
    def weighted_choice(self, choices):
        values, weights = zip(*choices)
        total = 0
        total_weights = []
        for w in weights:
            total += w
            total_weights.append(total)
        x = random.random() * total
        i = bisect(total_weights, x)
        return values[i]

            
############################################################
    # Calculate the optimal policy of given map position.
    # Deal with stochastic planning for each possible moves.
    # Calculate each value of moves by accounting every 
    # possible moves the robot may moves.
    def calculate_policy(self, i, j):

        moves_array = []
        max_value_move = []
        
        for index, move in enumerate(self.move_list):
            next_position_i = i + move[0]
            next_position_j = j + move[1]

            if next_position_i >= 0 and next_position_j >= 0 and next_position_i < len(self.map) and next_position_j < len(self.map[0]):
                if self.map[next_position_i][next_position_j] == self.map_dict["MAP_WALL"]:
                    moves_array.append([move, [i, j]])
                else:
                    moves_array.append([move, [next_position_i, next_position_j]])
            else:
                moves_array.append([move, [i, j]])
            
        for moves in moves_array:
            move_global = moves[0]
            total_value = 0
            for move_local, next_position in moves_array:
                probability = self.get_probability(move_global, move_local)
                reward = self.get_reward(i, j, next_position[0], next_position[1])
                total_value = total_value + probability * (reward + self.discount_factor * self.value_map[next_position[0]][next_position[1]])
            max_value_move.append([total_value, move_global])

        return sorted(max_value_move, reverse=True)[0]

############################################################
    # Calculate reward for moving into new position in the
    # map.
    def get_reward(self, i, j, next_position_i, next_position_j):
        
        if self.map[next_position_i][next_position_j] == self.map_dict["MAP_GOAL"]:
            return self.reward_for_reaching_goal + self.reward_for_each_step
        elif self.map[next_position_i][next_position_j] == self.map_dict["MAP_PIT"]:
            return self.reward_for_falling_in_pit + self.reward_for_each_step
        elif [i, j] == [next_position_i, next_position_j] or self.map[next_position_i][next_position_j] == self.map_dict["MAP_WALL"]:
            return self.reward_for_hitting_wall
        else:
            return self.reward_for_each_step

############################################################
    # Return the probability of robot moving in particular
    # direction depends on where it is currently going to move.
    def get_probability(self, move_global, move_local):
        
        if move_global == move_local:
            return self.prob_move_forward
        elif move_global == self.opposite_direction(move_local):
            return self.prob_move_backward
        elif move_global == self.left_direction(move_local):
            return self.prob_move_left
        else:
            return self.prob_move_right

############################################################
    # Explore helper method for epsilon greedy.
    def explore(self, epsilon):
        r = random.random()
        return r < epsilon

############################################################
    # Class functions to find how robot is moving locally
    # from its global position.
    def opposite_direction(self, move_local):

        if move_local == [0, 1]:
            return [0, -1]
        elif move_local == [0, -1]:
            return [0, 1]
        elif move_local == [1, 0]:
            return [-1, 0]
        elif move_local == [-1, 0]:
            return [1, 0]

    def left_direction(self, move_local):

        if move_local == [0, 1]:
            return [-1, 0]
        elif move_local == [0, -1]:
            return [1, 0]
        elif move_local == [1, 0]:
            return [0, 1]
        elif move_local == [-1, 0]:
            return [0, -1]

    def right_direction(self, move_local):

        if move_local == [0, 1]:
            return [1, 0]
        elif move_local == [0, -1]:
            return [-1, 0]
        elif move_local == [1, 0]:
            return [0, -1]
        elif move_local == [-1, 0]:
            return [0, 1]

############################################################
    # Class functions to convert map position into a string
    # list to be published to policy_list.
    def move_list_direction(self, move):

        if move == [0, 1]:
            return self.policy_dict["POLICY_EAST"]
        elif move == [0, -1]:
            return self.policy_dict["POLICY_WEST"]
        elif move == [1, 0]:
            return self.policy_dict["POLICY_SOUTH"]
        elif move == [-1, 0]:
            return self.policy_dict["POLICY_NORTH"]

    def direction_move_list(self, move):
        
        if move == self.policy_dict["POLICY_EAST"]:
            return [0, 1]
        elif move == self.policy_dict["POLICY_WEST"]:
            return [0, -1]
        elif move == self.policy_dict["POLICY_SOUTH"]:
            return [1, 0]
        elif move == self.policy_dict["POLICY_NORTH"]:
            return [-1, 0]

    def qvalue_move_list(self, move):
        
        if move == self.qvalue_dict["QVALUE_EAST"]:
            return [0, 1]
        elif move == self.qvalue_dict["QVALUE_WEST"]:
            return [0, -1]
        elif move == self.qvalue_dict["QVALUE_NORTH"]:
            return [-1, 0]
        elif move == self.qvalue_dict["QVALUE_SOUTH"]:
            return [1, 0]

    def move_list_qvalue(self, move):
        
        if move == [0, 1]:
            return self.qvalue_dict["QVALUE_EAST"]
        elif move == [0, -1]:
            return self.qvalue_dict["QVALUE_WEST"]
        elif move == [-1, 0]:
            return self.qvalue_dict["QVALUE_NORTH"]
        elif move == [1, 0]:
            return self.qvalue_dict["QVALUE_SOUTH"]

    def qvalue_policy(self, move):

        if move == 0:
            return self.policy_dict["POLICY_EAST"]
        elif move == 1:
            return self.policy_dict["POLICY_WEST"]
        elif move == 2:
            return self.policy_dict["POLICY_NORTH"]
        elif move == 3:
            return self.policy_dict["POLICY_SOUTH"]


if __name__ == "__main__":
    
    try:
        md = qlearning()
    except rospy.ROSInterruptException:
        pass
