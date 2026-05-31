import mesa as ms
import numpy as np

class Agent(ms.Agent):
    def __init__(self, model, age):
        super().__init__(model)
        self.age = age
        self.center = np.array([0.,0.])

    def cluster_center(self):
        coords = [a.pos for a in self.model.agents]
        self.center = np.array([np.mean(coords[:][0:1]), np.mean(coords[:][1:2])])

    def step(self, speed, end_point):
        #mean = self.model.agents.agg("pos[0]", np.mean)
        #vec = end_point - mean
        vec = self.model.grid.get_heading(end_point, self.center)
        self.pos = self.pos + vec * speed
        #print(f"Agent {self.unique_id} now is {self.pos}")
        #self.age += 1
        #print(f"Agent {self.unique_id} now is {self.age} years old")
        # Whatever else the agent does when activated