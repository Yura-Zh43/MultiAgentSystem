import numpy as np
import itertools

class Walls():

    def __init__(self, points):
        self.points = points
        points_T = [np.transpose(wall) for wall in points]
        self.X = [wall[0] for wall in points_T]
        self.Y = [wall[1] for wall in points_T]
        
        separators = [len(wall) for wall in points]
        for i in range(1, len(separators)):
            separators[i] = separators[i] + separators[i - 1]

        self.normal_vectors = np.empty([sum(len(wall) - 1 for wall in points), 2])
        self.walls = np.empty([len(self.normal_vectors), 2, 2])

        self.lengths2 = np.empty(len(self.walls))
        self.A = np.empty(len(self.walls))
        self.B = np.empty(len(self.walls))
        self.C = np.empty(len(self.walls))

        k = 0
        for i, wall in enumerate(self.points):
            for j in range(len(wall) - 1):
                self.normal_vectors[k] = [wall[j + 1][1] - wall[j][1], wall[j][0] - wall[j + 1][0]]
                self.walls[k] = self.points[i][j:j+2]

                self.A[k] = self.walls[k][1][1] - self.walls[k][0][1]
                self.B[k] = self.walls[k][0][0] - self.walls[k][1][0]
                self.C[k] = self.walls[k][0][1] * self.walls[k][1][0] - \
                    self.walls[k][0][0] * self.walls[k][1][1]
                self.lengths2[k] = self.A[k]**2 + self.B[k]**2

                k = k + 1

        self.normal_vectors_T = np.transpose(self.normal_vectors)

        '''
        self.dot_prods = np.empty()
        self.update_data()

        self.temp_normal_vectors = np.empty()
        self.dot_prods = np.empty()
        self.dists = np.empty()
        self.update_data()'''

    def get_normal_vectors(self, pos, d, radius, is_sticking, eps=1e-2):
        if is_sticking:
            radius2 = (radius + eps)**2
        else:
            radius2 = radius**2

        ans = np.empty([0, 2])
        wall_inds = np.empty(0, dtype=int)
        dot_prods = d @ self.normal_vectors_T
        dot_prods[np.abs(dot_prods) < 1e-8] = 0

        for i, x in enumerate(dot_prods):
        #for i in range(len(self.normal_vectors)):
            if x <= 0 and np.dot(pos - self.walls[i][0], self.normal_vectors_T[:,i]) >= 0:
            #if np.dot(pos - self.walls[i][0], self.normal_vectors_T[:,i]) >= 0:
                vecs = pos - self.walls[i]
                dist1 = np.dot(vecs[0], vecs[0])
                dist2 = np.dot(vecs[1], vecs[1])
                inds = np.argsort([dist1, dist2, self.lengths2[i]])

                if inds[2] == 2:
                    if (self.A[i] * pos[0] + self.B[i] * pos[1] + self.C[i])**2 / self.lengths2[i] <= radius2:
                            ans = np.append(ans, self.normal_vectors[i:i+1], axis=0)
                            wall_inds = np.append(wall_inds, i)
                elif inds[2] == 1:
                    if dist2 < dist1 + self.lengths2[i]:
                        if (self.A[i] * pos[0] + self.B[i] * pos[1] + self.C[i])**2 / self.lengths2[i] <= radius2:
                            ans = np.append(ans, self.normal_vectors[i:i+1], axis=0)
                            wall_inds = np.append(wall_inds, i)
                    else:
                        if dist1 <= radius2:
                            ans = np.append(ans, vecs[0:1], axis=0)
                            wall_inds = np.append(wall_inds, i)
                else:
                    if dist1 < dist2 + self.lengths2[i]:
                        if (self.A[i] * pos[0] + self.B[i] * pos[1] + self.C[i])**2 / self.lengths2[i] <= radius2:
                            ans = np.append(ans, self.normal_vectors[i:i+1], axis=0)
                            wall_inds = np.append(wall_inds, i)
                    else:
                        if dist2 <= radius2:
                            ans = np.append(ans, vecs[1:2], axis=0)
                            wall_inds = np.append(wall_inds, i)

        return ans, wall_inds

    def normal_vector(self, pos, radius, wall_ind):
        i = wall_ind
        radius2 = radius**2

        vecs = pos - self.walls[i]
        dist1 = np.dot(vecs[0], vecs[0])
        dist2 = np.dot(vecs[1], vecs[1])
        inds = np.argsort([dist1, dist2, self.lengths2[i]])

        if inds[2] == 2:
            #if (self.A[i] * pos[0] + self.B[i] * pos[1] + self.C[i])**2 / self.lengths2[i] <= radius2:
            return self.normal_vectors[i:i+1][0]
        elif inds[2] == 1:
            if dist2 < dist1 + self.lengths2[i]:
                #if (self.A[i] * pos[0] + self.B[i] * pos[1] + self.C[i])**2 / self.lengths2[i] <= radius2:
                return self.normal_vectors[i:i+1][0]
            else:
                #if dist1 <= radius2:
                return vecs[0:1][0]
        else:
            if dist1 < dist2 + self.lengths2[i]:
                #if (self.A[i] * pos[0] + self.B[i] * pos[1] + self.C[i])**2 / self.lengths2[i] <= radius2:
                return self.normal_vectors[i:i+1][0]
            else:
                #if dist2 <= radius2:
                return vecs[1:2][0]

    #def calc_dot_prods(self, direction):
    #    dot_prods = direction @ self.normal_vectors

    #def update_data(self):
    #    self.calc_dot_prods()

    #def get_data(self, coords, repulsion_radius, viewing_radius):
    #    return dirs