import mujoco as mj
import numpy as np
import os

class CDPRSimulation:
    def __init__(self, model_path):
        self.model = mj.MjModel.from_xml_path(model_path)
        self.data = mj.MjData(self.model)
        self._setup_simulation()
        
    def _setup_simulation(self):
        """Initialize simulation parameters"""
        self.simend = 100
        self.threshold = 0.002
        self.received_goal = False
        
    def reset(self):
        mj.mj_resetData(self.model, self.data)
        mj.mj_forward(self.model, self.data)
        
    def step(self):
        mj.mj_step(self.model, self.data)