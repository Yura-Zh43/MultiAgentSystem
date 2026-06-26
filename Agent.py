from asyncio.windows_events import NULL
import math as m
import mesa as ms
import numpy as np

def normalization(vec):
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return np.array([0, 0])
    else:
        return vec / norm

class Agent(ms.Agent):
    def __init__(self, model):
        super().__init__(model)
        self.gr = self.model.grid
        self.next_pos = np.array((0.,0.))

        self.dir = np.array((0.,0.))
        self.next_dir = np.array((0.,0.))

        #self.cl_dir = np.array((0.,0.))
        #self.next_cl_dir = np.array((0.,0.))

        self.d = np.array((0.,0.))
        self.next_d = np.array((0.,0.))

        self.is_sticking = False
        self.next_is_sticking = False
        self.num_of_meetings = 0
        self.next_num_of_meetings = 0
        self.wall_ind = 0
        self.next_wall_ind = 0
        self.is_counterclockwise = False
        self.next_is_counterclockwise = False


    def swap_data(self):
        self.pos, self.next_pos = self.next_pos, self.pos
        self.dir, self.next_dir = self.next_dir, self.dir
        self.d, self.next_d = self.next_d, self.d

        self.num_of_meetings, self.next_num_of_meetings = self.next_num_of_meetings, self.num_of_meetings
        self.is_sticking, self.next_is_sticking = self.next_is_sticking, self.is_sticking
        self.wall_ind, self.next_wall_ind = self.next_wall_ind, self.wall_ind
        self.is_counterclockwise, self.next_is_counterclockwise =\
            self.next_is_counterclockwise, self.is_counterclockwise

    def neighbors(self, agents, radius, include_center=False):
        pos = agents.get('pos')
        if len(pos) > 0:
            deltas = pos - self.pos
            dists = deltas[:, 0] ** 2 + deltas[:, 1] ** 2
            (idxs,) = np.where(dists <= radius**2)
            return ms.agent.AgentSet([agents[x] for x in idxs if include_center or dists[x] > 0], random=self.model.random)
        else:
            return ms.agent.AgentSet([], random=self.model.random)

    def nearest_agent_id(self, agents):
        if len(agents) > 0:
            pos = agents.get('pos')
            deltas = pos - self.pos
            return np.argmin(deltas[:, 0] ** 2 + deltas[:, 1] ** 2)
        else:
            return None

    def preferred_dir(self):
        ans = np.array((0.,0.))
        sticking_dir = -self.model.walls.normal_vector(self.pos, self.model.min_radius, self.wall_ind)
        sticking_dir = normalization(sticking_dir)
        if self.is_counterclockwise:
            ans[0] = sticking_dir[1]
            ans[1] = -sticking_dir[0]
        else:
            ans[0] = -sticking_dir[1]
            ans[1] = sticking_dir[0]
        return ans

    def rand_dir(self):
        ans = np.array((1.,0.))
        self.theta = self.model.random.uniform(0, 2*m.pi)
        R = np.array(((np.cos(self.theta), -np.sin(self.theta)), (np.sin(self.theta), np.cos(self.theta))))
        return R @ ans

    def adjustment_dir(self, dir_, normals, agents):
        normals_T = np.transpose(normals)
        dot_prods = dir_ @ normals_T
        dot_prods[np.abs(dot_prods) < 1e-8] = 0

        if self.next_is_sticking:
            for i in range(len(dot_prods)):
                if dot_prods[i] < 0:
                    return np.array([0, 0])
            return dir_
        else:
            vecs = np.empty([0, 2])
            for i in range(len(dot_prods)):
                if dot_prods[i] < 0:
                    vecs = np.append(vecs, [[-normals[i][1], normals[i][0]]], axis=0)

            if len(vecs) > 0:
                '''
                if len(agents) > 0:
                    mean_dir = np.mean(agents.get('dir'), axis=0)
                    if mean_dir[0]**2 + mean_dir[1]**2 > 0:
                        dir_ = mean_dir
                        dir_ = normalization(dir_)
                #'''

                ans = np.array([0, 0])

                dot_prods = vecs @ normals_T
                dot_prods[np.abs(dot_prods) < 1e-8] = 0
                signs = np.sign(dot_prods)

                arr = [np.unique(line) for line in signs]
                dot_prod_cur = 0
                dot_prod = 0
                for i in range(len(arr)):
                    if len(arr[i]) < 3:
                        if np.array_equal(arr[i], [-1, 1]):
                            break
                        elif arr[i][0] == -1:
                            vecs[i] = -vecs[i]
                            dot_prod_cur = np.dot(dir_, vecs[i])
                        elif arr[i][-1] == 1:
                            dot_prod_cur = np.dot(dir_, vecs[i])
                        else:
                            dots = np.dot(dir_, vecs[i]), np.dot(dir_, -vecs[i])
                            if dots[0] < dots[1]:
                                dot_prod_cur = dots[1]
                                vecs[i] = -vecs[i]
                            else:
                                dot_prod_cur = dots[0]

                        if dot_prod < dot_prod_cur:
                            ans = vecs[i]
                            dot_prod = dot_prod_cur
                return ans
            else:
                return dir_

    def sticking(self, normals, wall_inds, sticking_agents):
        if len(normals) > 0:
            '''normals_T = np.transpose(normals)
            #normals_T = self.model.walls.normal_vectors_T[:, wall_inds]
            dot_prods = dir_ @ normals_T
            dot_prods[np.abs(dot_prods) < 1e-8] = 0

            ind = np.argmin(dot_prods)'''
            dir_to_end = self.gr.get_heading(self.pos, self.model.end_point)
            normals_T = np.transpose(normals)
            #normals_T = self.model.walls.normal_vectors_T[:, wall_inds]
            dot_prods = dir_to_end @ normals_T
            dot_prods[np.abs(dot_prods) < 1e-8] = 0

            ind = np.argmin(dot_prods)

            #if dot_prods[ind] <= 0:
            sticking_dir = -normals[ind]
            if np.dot(sticking_dir, dir_to_end) > 0:
                self.next_is_sticking = True
                self.next_wall_ind = wall_inds[ind]
                if not self.is_sticking:
                    #sticking_agents.select(lambda ag: self.next_wall_ind == ag.wall_ind, inplace=True)
                    #if len(sticking_agents) > 0:
                    #    nearest_agent_id = self.nearest_agent_id(sticking_agents)
                    #    self.next_is_counterclockwise = sticking_agents[nearest_agent_id].is_counterclockwise
                    #    self.next_num_of_meetings = sticking_agents[nearest_agent_id].num_of_meetings
                    #elif np.cross(sticking_dir, dir_to_end) < 0:
                    if np.cross(sticking_dir, dir_to_end) < 0:
                        self.next_is_counterclockwise = True
                    else:
                        self.next_is_counterclockwise = False
            else:
                self.next_is_sticking = False
                self.next_num_of_meetings = 0
            #else:
            #    self.next_is_sticking = False
            #    self.next_num_of_meetings = 0
        else:
            self.next_is_sticking = False
            self.next_num_of_meetings = 0

        '''if self.next_is_sticking:
            if self.next_is_counterclockwise:
                dir_[0] = sticking_dir[1]
                dir_[1] = -sticking_dir[0]
            else:
                dir_[0] = -sticking_dir[1]
                dir_[1] = sticking_dir[0]

        return dir_'''

    def step(self):
        if self.next_is_counterclockwise != self.is_counterclockwise:
            self.next_is_counterclockwise = self.is_counterclockwise

        if self.next_num_of_meetings != self.num_of_meetings:
            self.next_num_of_meetings = self.num_of_meetings

        visible_agents = self.neighbors(self.gr.agents, self.model.viewing_radius)
        nearby_agents = self.neighbors(self.gr.agents, self.model.min_radius)

        normals, wall_inds = self.model.walls.get_normal_vectors(self.pos, self.d, self.model.min_radius, self.is_sticking)
        if len(normals) > 0:
            normals = np.apply_along_axis(lambda d: normalization(d), axis=1, arr=normals)
        #sticking_agents = nearby_agents.select(lambda ag: ag.is_sticking == True)
        sticking_agents = visible_agents.select(lambda ag: ag.is_sticking == True)
        self.sticking(normals, wall_inds, sticking_agents)

        #if self.is_sticking:
        if self.next_is_sticking:
            #self.next_d = self.preferred_dir()
            sticking_dir = -self.model.walls.normal_vector(self.pos, self.model.min_radius, self.next_wall_ind)
            sticking_dir = normalization(sticking_dir)
            if self.next_is_counterclockwise:
                self.next_d[0] = sticking_dir[1]
                self.next_d[1] = -sticking_dir[0]
            else:
                self.next_d[0] = -sticking_dir[1]
                self.next_d[1] = sticking_dir[0]
        elif np.array_equal(self.dir, [0., 0.]):
            self.next_d = self.rand_dir()
            self.next_d = normalization(self.next_d)
        else:
            self.next_d = self.gr.get_heading(self.pos, self.model.end_point)
            if self.next_d[0]**2 + self.next_d[1]**2 > 0:
                self.next_d = normalization(self.next_d)
            else:
                self.next_d = np.array((0.,0.))

            if len(visible_agents) > 0:
                #mean_dir = np.mean(visible_agents.get('dir'), axis=0)
                mean_dir = np.mean(visible_agents.map(lambda ag: ag.d if ag.is_sticking else ag.dir), axis=0)
                if mean_dir[0]**2 + mean_dir[1]**2 > 0:
                    mean_dir = normalization(mean_dir)
                    self.next_d = (1 - self.model.coef_self_dir) * mean_dir + self.model.coef_self_dir * self.next_d
                    if self.next_d[0]**2 + self.next_d[1]**2 > 0:
                        self.next_d = normalization(self.next_d)
                    else:
                        self.next_d = np.array((0.,0.))

        #sticking_agents = nearby_agents.select(lambda ag: ag.is_sticking == True)
        #if self.is_sticking:
        if self.next_is_sticking:
            sticking_agents2 = sticking_agents.select(lambda ag: ag.num_of_meetings > self.num_of_meetings and\
                np.dot(self.next_d, ag.pos - self.pos) > 0 and np.dot(ag.d, ag.pos - self.pos) < 0)
                #ag.is_counterclockwise != self.is_counterclockwise)
            sticking_agents3 = sticking_agents.select(lambda ag: ag.num_of_meetings == self.num_of_meetings and\
                np.dot(self.next_d, ag.pos - self.pos) > 0 and np.dot(ag.d, ag.pos - self.pos) < 0)
                #ag.is_counterclockwise != self.is_counterclockwise)
            sticking_agents4 = sticking_agents.select(lambda ag: ag.num_of_meetings < self.num_of_meetings and\
                np.dot(self.next_d, ag.pos - self.pos) > 0 and np.dot(ag.d, ag.pos - self.pos) < 0)
                #ag.is_counterclockwise != self.is_counterclockwise)
            if len(sticking_agents2) > 0:
                nearest_agent_id = self.nearest_agent_id(sticking_agents2)
                self.next_num_of_meetings = 1 + sticking_agents2[nearest_agent_id].num_of_meetings
                #self.next_is_counterclockwise = sticking_agents2[nearest_agent_id].is_counterclockwise
                self.next_is_counterclockwise = not self.next_is_counterclockwise
                self.next_d = -self.next_d
            elif len(sticking_agents3) > 0:
                nearest_agent_id = self.nearest_agent_id(sticking_agents3)
                self.next_num_of_meetings = self.next_num_of_meetings + 1
                if sticking_agents3[nearest_agent_id].unique_id < self.unique_id:
                    #self.next_is_counterclockwise = sticking_agents3[nearest_agent_id].is_counterclockwise
                    self.next_is_counterclockwise = not self.next_is_counterclockwise
                    self.next_d = -self.next_d
            elif len(sticking_agents4) > 0:
                self.next_num_of_meetings = 1 + self.num_of_meetings
            else:
                self.next_num_of_meetings = 1 + self.num_of_meetings

        #normals, wall_inds = self.model.walls.get_normal_vectors(self.pos, self.model.min_radius, self.is_sticking)
        #if len(normals) > 0:
        #    normals = np.apply_along_axis(lambda d: normalization(d), axis=1, arr=normals)
        #self.next_dir = self.sticking(self.next_dir, normals, wall_inds, sticking_agents)

        '''
        if len(normals) > 0:
            normals_T = self.model.walls.normal_vectors_T[:, wall_inds]
            dot_prods = self.next_dir @ normals_T
            dot_prods[np.abs(dot_prods) < 1e-8] = 0
            d_ind = np.empty(0, dtype=int)
            for i, x in enumerate(dot_prods):
                if x > 0:
                    d_ind = np.append(d_ind, i)
            if len(d_ind) > 0:
                normals = np.delete(normals, d_ind, axis=0)
                wall_inds = np.delete(wall_inds, d_ind)
        #'''

        if len(nearby_agents) > 0:
            normals = np.append(normals, self.gr.get_heading(nearby_agents.get('pos'), self.pos), axis=0)
        if len(normals) > 0:
            normals = np.apply_along_axis(lambda d: normalization(d), axis=1, arr=normals)
            self.next_dir = self.adjustment_dir(self.next_d, normals, visible_agents)
        else:
            self.next_dir = self.next_d

        self.next_pos = self.pos + self.next_dir * self.model.speed
        #self.gr.move_agent(self, self.pos + self.next_dir * self.model.speed)