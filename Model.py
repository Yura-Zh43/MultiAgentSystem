import mesa as ms
import math as m
import numpy as np
import threading as th
import Agent
import Walls
from sklearn import cluster as cl

class Model(ms.Model):
    def __init__(self, n_agents, points, n_threads, seed=None):
        super().__init__(seed=seed)

        self.is_end = False
        self.is_end1 = False

        self.grid = ms.space.ContinuousSpace(20, 10, torus=False)

        #self.repulsion_radius = 0.3
        #self.viewing_radius = self.repulsion_radius * 2
        self.min_radius = 0.2
        self.repulsion_radius = self.min_radius * 2

        self.viewing_radius = self.min_radius * 4
        #self.viewing_radius = 100
        #self.viewing_radius = 0.1

        self.speed = 0.01
        #self.gamma = 0.2

        self.coef_self_dir = 0.1#0.5

        #'''
        self.min_radius = 0.2#0.1 0.2 0.3
        self.viewing_radius = 0.7#0.6 0.7 0.8
        self.coef_self_dir = 0.5#0.1 0.3 0.5 0.7 0.9
        #'''
        '''
        self.min_radius = 0.3#0.1 0.2 0.3
        self.viewing_radius = 0.7#0.6 0.7 0.8
        self.coef_self_dir = 0.8#0.8 0.9 1.0
        #'''

        #self.end_point = np.array((2., 2.))
        self.end_point = np.array((2., 5.))

        self.clusters = np.empty(0)
        self.cluster_centers = []

        self.walls = Walls.Walls(points)

        self.theta = m.pi/180
        self.R_m = np.array(((np.cos(-self.theta), -np.sin(-self.theta)), (np.sin(-self.theta), np.cos(-self.theta))))
        self.R_p = np.array(((np.cos(self.theta), -np.sin(self.theta)), (np.sin(self.theta), np.cos(self.theta))))

        for i in range(n_agents):
            a = Agent.Agent(self)
            coords = np.array((self.random.uniform(17, 19), self.random.uniform(2, 8)))
            #if i % 2 == 0:
            #    coords = np.array((self.random.uniform(9, 10), self.random.uniform(5, 7)))
            #else:
            #    coords = np.array((self.random.uniform(6, 8), self.random.uniform(9, 10)))
            a.dir = self.grid.get_heading(coords, self.end_point)
            a.dir = a.dir / np.linalg.norm(a.dir)
            a.d = a.dir
            self.grid.place_agent(a, coords)

        '''
        self.processes = []
        self.barrier = th.Barrier(n_threads + 1)
        self.is_finished = False
        a, b = divmod(n_agents, n_threads)
        sizes = [a + 1] * b + [a] * (n_threads - b)
        ind = 0
        for i in range(n_threads):
            self.processes.append(th.Thread(target=Model.step_proc, args=(\
                ms.agent.AgentSet(self.agents[ind:ind + sizes[i]], random=self.random),\
                self.barrier,self.is_finished,)))
            self.processes[-1].start()
            ind = ind + sizes[i]
        #list(self.grid._agent_to_index)
        #'''

        self.clustering()

        #self.distances_between_agents
        #self.calc_distances()

    def step_proc(agents, barrier, is_finished):
        while not is_finished:
            agents.do("step")
            barrier.wait()
            barrier.wait()

    def clustering(self):
        dbscan = cl.DBSCAN(eps=self.viewing_radius, min_samples=2).fit(self.grid.agents.get('pos'))
        self.clusters = dbscan.labels_
        mat = np.insert(np.transpose(self.clusters[np.newaxis]), 1, np.arange(len(self.clusters)), axis=1)
        mat = mat[mat[:, 0].argsort()]
        if mat[0,0] == -1:
            mat = np.split(mat[:,1], np.unique(mat[:, 0], return_index=True)[1][1:])[1:]
        else:
            mat = np.split(mat[:,1], np.unique(mat[:, 0], return_index=True)[1][1:])
        pos = np.array(self.grid.agents.get('pos'))
        for i in range(len(mat)):
            mat[i] = pos[mat[i]]
        self.cluster_centers = np.empty([0, 2])
        for coords in mat:
            self.cluster_centers = np.append(self.cluster_centers, [np.mean(coords, axis=0)], axis=0)

    #def calc_distances(self):

    def step(self):
        '''
        self.barrier.wait()

        self.agents.do("swap_data")
        self.clustering()
        if self.grid.get_distance(np.mean(self.grid.agents.get('pos'), axis=0),\
            self.end_point) < 2 * self.speed:
            self.end_point = np.array((self.random.uniform(1, 9), self.random.uniform(1, 9)))

        self.barrier.wait()
        #'''
        #'''
        self.agents.do("step")

        self.agents.do("swap_data")
        self.clustering()
        #if self.grid.get_distance(np.mean(self.grid.agents.get('pos'), axis=0),\
        #    self.end_point) < 2 * self.speed:
        #    self.end_point = np.array((self.random.uniform(1, 9), self.random.uniform(1, 9)))

        agents = self.neighbors(self.end_point, self.grid.agents, np.sqrt(len(self.grid.agents)) * 2 * self.min_radius, True)
        agents1 = self.neighbors(self.end_point, self.grid.agents, np.sqrt(len(self.grid.agents)) * self.min_radius, True)
        #if self.grid.get_distance(np.mean(self.grid.agents.get('pos'), axis=0),\
        #    self.end_point) < self.min_radius and len(self.grid.agents) == len(agents):
        if not self.is_end and len(self.grid.agents) == len(agents):
            print(self.steps)
            self.is_end = True

        if len(self.grid.agents) == len(agents1):
            print(self.steps)
            self.is_end1 = True
            self.end_point = np.array((self.random.uniform(1, 9), self.random.uniform(1, 9)))
        #'''

    def neighbors(self, pos, agents, radius, include_center=False):
        ag_pos = agents.get('pos')
        if len(agents) > 0:
            deltas = ag_pos - pos
            dists = deltas[:, 0] ** 2 + deltas[:, 1] ** 2
            (idxs,) = np.where(dists <= radius**2)
            return ms.agent.AgentSet([agents[x] for x in idxs if include_center or dists[x] > 0], random=self.random)
        else:
            return ms.agent.AgentSet([], random=self.model.random)