import mesa as ms
import numpy as np
from Agent import Agent

class Model(ms.Model):
    def __init__(self, n_agents):
        super().__init__()
        self.grid = ms.space.ContinuousSpace(10, 10, torus=True)
        self.speed = 0.01
        self.end_point = np.array([1., 1.])

        for _ in range(n_agents):
            initial_age = self.random.randint(0, 80)
            a = Agent(self, initial_age)
            coords = np.array([self.random.uniform(9, 10), self.random.uniform(9, 10)])
            self.grid.place_agent(a, coords)

    def step(self):
        self.agents.do("cluster_center")
        self.agents.do("step", self.speed, self.end_point)