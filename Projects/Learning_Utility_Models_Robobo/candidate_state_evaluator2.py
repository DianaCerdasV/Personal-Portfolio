"""
The shiny, all new, MDB 3.0.

Available from (we are still thinking about this...)
Distributed under the (yes, we are still thinking about this too...).
"""

import math
import random
import numpy as np

#from action_chooser import ActionChooser
#from back_prop import *
# from forward_model import ForwardModel
#from forward_model import ForwardModel
#from utility_models import UtilityModel
from scipy.spatial import distance



class CandidateStateEvaluator():
    def __init__(self):
        #self.forward_model = ForwardModel()
        #self.action_chooser = ActionChooser()
        

        # Variables to control the Brownian motion (intrinsic motivation)
        self.n_random_steps = 0
        self.max_random_steps = 3

        self.intrinsic_exploration_type = 'Novelty'  # 'Brownian' or 'Novelty'

        self.n = 0.5  # Coefficient that regulates the balance between the relevance of distant and near states

    def get_action(self, exploration_type, candidate_actions, intrinsic_memory):

        if exploration_type == 'Int':  # Intrinsic Motivation
            if self.intrinsic_exploration_type == 'Brownian':
                # Brownian motion
                self.n_random_steps += 1
                if self.n_random_steps > self.max_random_steps:
                    action = np.random.uniform(-45, 45)
                    self.max_random_steps = np.random.randint(1, 4)
                    self.n_random_steps = 0
                else:
                    action = 0
            elif self.intrinsic_exploration_type == 'Novelty':
                #candidate_actions = self.action_chooser.get_candidate_actions()
                candidates_eval = self.get_novelty_evaluation(candidate_actions, intrinsic_memory)
                #action = self.action_chooser.choose_action(candidates_eval)
                action = list(candidates_eval[-1])  # Convert into
                #print("Hola", action)

        return action, candidates_eval



    def get_novelty_evaluation(self, candidates, trajectory_buffer):
        """Return the list of candidates actions sorted according to their novelty value

        :param sim_data: 
        :param candidates: list o candidate actions
        :param trajectory_buffer: buffer that stores the last perceptual states the robot has experienced
        :return: list of candidates actions sorted according to its novelty valuation
        """
        #print("1", candidates)
        evaluated_candidates = []
        candidatesP = []
        for i in range(len(candidates)):
            candidatesP = candidates[i][:]
            candidatesP.pop(2)
            #print("Candidatos", candidates[i], candidatesP)
            valuation = self.get_novelty(candidatesP, trajectory_buffer)
            evaluated_candidates.append((candidates[i],) + (valuation,))
            #print("Candidatos:", evaluated_candidates)
            #evaluated_candidates = candidates[i] + [valuation]

        # Ordenor los estados evaluados
        evaluated_candidates.sort(key=lambda x: x[-1])

        return evaluated_candidates

    def get_novelty(self, candidate_state, trajectory_buffer):
        """Return the novelty for each individual candidate

        :param sim_data: 
        :param candidate_action: 
        :param trajectory_buffer: buffer that stores the last perceptual states the robot has experienced
        :return: novelty of the candidate state
        """
        # Creo que esta es (dist_red, dist_green)
        #candidate_state = self.forward_model.predicted_state(candidate_action, sim_data)
        novelty = 0
        for i in range(len(trajectory_buffer)):
            novelty += pow(distance.euclidean(candidate_state, trajectory_buffer[i]), self.n)
            #print(novelty)

        novelty = novelty / len(trajectory_buffer)

        return novelty

