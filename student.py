from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

class StudentBot(BaseAgent):
    def initialize_agent(self):
        pass
    
    def get_output(self, packet):
        