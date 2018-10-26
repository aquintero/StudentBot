from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from game import Game
from dqn import DQN
from model import Model
import numpy as np

class StudentBot(BaseAgent):
    def initialize_agent(self):
        dqn = DQN()
        self.model = Model(dqn)
        self.game = Game()
        self.epsilon = 0.05
    
    def get_output(self, packet):
        state, reward, next_state = self.game.step(packet)
        action = game.sample_action()
        if np.random.random() > self.epsilon:
            action = self.model.predict(game.state)
        