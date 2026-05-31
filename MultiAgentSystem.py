# -*- coding: cp1251 -*-
#from tkinter import W
import mesa as ms
import time as t
#import solara as s
from Model import Model
from mesa.visualization.components import AgentPortrayalStyle

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.backend_bases
import numpy as np
import time

def color_map(max_depth):
    colormap = []

    if max_depth > 0:
        offsets = [1, 0.5, 0.75, 0.25]
        values = [0.99, 0.75, 0.5]
        value_ind = 0
        for offset in offsets:
            colormap.append(mcolors.hsv_to_rgb([offset, 1, values[value_ind]]))
            value_ind = (value_ind + 1) % 3

        cur_offset = 0.125
        prev_offset = 0.25
        for depth in range(max_depth - 1):
            temp = cur_offset
            for _ in range(2**depth):
                for offset in offsets:
                    colormap.append(mcolors.hsv_to_rgb([offset - temp, 1, values[value_ind]]))
                    value_ind = (value_ind + 1) % 3
                temp = temp + prev_offset
            prev_offset = cur_offset
            cur_offset = cur_offset / 2

    colormap.append(np.array([0,0,0]))

    return colormap

def onKeyEvent(event: matplotlib.backend_bases.KeyEvent) -> None:
    global is_pause
    if event.key == ' ':
        is_pause = not is_pause
    elif event.key == 'enter':
        if is_pause:
            global next_frame
            next_frame = True
    elif event.key == 'r':
        global model
        #model = Model(n_agents)
        model = Model(n_agents, points, n_threads, seed)
        global i
        i = 0
        global colormap
        colormap = [palette[cl] for cl in model.clusters]

def reDraw(i, model, fig, lines, circles):
    ax.set_title(f'Шаг: {i}', loc='left')

    pos = [ag.pos for ag in model.grid.agents]

    lines[0].set_offsets(pos)
    lines[0].set_color(colormap)

    lines[1].set_offsets(model.end_point)

    #lines[2].set_offsets(model.grid.agents.get('center'))

    #lines[3].set_offsets(np.mean(model.grid.agents.get('pos'), axis=0))
    #lines[3].set_offsets(model.cluster_centers)

    #circles[0].set_center(model.end_point)
    circles[1].set_center(model.end_point)

    fig.canvas.draw()
    #plt.pause(0.01)

is_pause = True
next_frame = False

#print(np.arctan2([1.], [0.]))

#points = [np.array([[0, 0],[0, 10],[10, 10],[10, 0],[0, 0]]), np.array([[5, 3],[5, 5],[7, 5],[5, 5],[5, 7],[5, 5],[3, 5],[5, 5],[5, 3]])]
#points = [np.array([[0, 0],[0, 10],[10, 10],[10, 0],[0, 0]]), np.array([[4, 4],[6, 4],[6, 6],[4, 6],[4, 4]])]
#points = [np.array([[0, 0],[0, 10],[15, 10],[15, 0],[0, 0]]), np.array([[7, 3],[8, 4],[4, 8],[3, 7],[7, 3]])]
points = [np.array([[0, 0],[0, 10],[20, 10],[20, 0],[0, 0]]),\
    np.array([[15, 2.5],[16, 4.5],[15, 4.5],[15, 2.5]]),\
    np.array([[15, 5.5],[16, 5.5],[15, 7.5],[15, 5.5]]),\
    np.array([[12, 1.5],[13, 1.5],[12, 2.5],[12, 1.5]]),\
    np.array([[12, 4],[13, 4],[13, 6],[12, 6],[12, 4]]),\
    np.array([[12, 7.5],[13, 8.5],[12, 8.5],[12, 7.5]]),\
    np.array([[9, 1],[10, 1],[10, 4],[9, 4],[9, 1]]),\
    np.array([[9, 6],[10, 6],[10, 9],[9, 9],[9, 6]])]
#points = [np.array([[0, 0],[0, 10],[20, 10],[20, 0],[0, 0]]), np.array([[14, 2],[15.5, 2],[16, 5],[15.5, 8],[14, 8],[14, 2]])]
#points = [np.array([[0, 0],[0, 10],[20, 10],[20, 0],[0, 0]]), np.array([[14, 4.5],[15, 5],[14, 5.5],[14, 4.5]])]
#points = [np.array([[0, 0],[0, 10],[10, 10],[10, 0],[0, 0]]), np.array([[6, 2],[7, 3],[8, 5],[6, 4],[5, 3],[6, 2]]), np.array([[5, 5],[6, 8],[3, 8],[3, 5],[5, 5]])]
n_agents = 40
n_threads = 2
seed = 1
#seed = 10
model = Model(n_agents, points, n_threads, seed)

#print(model.walls.get_normal_vectors(np.array([5.2,4.8]), np.array([-1,1])))

x, y = zip(*model.grid.agents.get('pos'))
centers_x, centers_y = zip(*model.grid.agents.get('center'))
global_center = np.mean(model.grid.agents.get('pos'), axis=0)

palette = color_map(5)
colormap = [palette[cl] for cl in model.clusters]

plt.ion()

fig = plt.figure(figsize=(14, 7))
fig.canvas.manager.window.wm_geometry("+%d+%d" % (480, 70))
ax = plt.axes(xlim=(0, 20), ylim=(0, 10))
ax.set_title('Шаг: 0', loc='left')
ax.set_aspect('equal')
line1 = ax.scatter(x, y, c=colormap)
line2 = ax.scatter(model.end_point[0], model.end_point[1], c='black', marker='x', s=30)
line3 = None#line3 = ax.scatter(centers_x, centers_y, facecolors='none', edgecolors='black')
line4 = None#line4 = ax.scatter(global_center[0], global_center[1], c='black', marker='+', s=30)

circle1 = None#circle1 = plt.Circle(model.end_point, model.min_radius, color='black', fill=False)
#ax.add_patch(circle1)
circle2 = plt.Circle(model.end_point, np.sqrt(n_agents) * model.min_radius, color='black', fill=False)
ax.add_patch(circle2)

walls = []
for i in range(len(model.walls.X)):
    walls.append(ax.plot(model.walls.X[i], model.walls.Y[i], color='black'))

key_press_event_id = fig.canvas.mpl_connect('key_press_event', onKeyEvent)

i = 0
all_time = 0
while True:
    #start_time = t.time()
    
    #if i % 3 == 0 or is_pause:
    if i % 50 == 0 or is_pause:
        fig.canvas.flush_events()
        reDraw(i, model, fig, (line1,line2,line3,line4), (circle1, circle2))
        #plt.savefig(f'img/{int(i/3)}.png')
    if not is_pause or is_pause and next_frame:
        '''ax.set_title(f'Шаг: {i}', loc='left')

        pos = [ag.pos for ag in model.grid.agents]

        line1.set_offsets(pos)
        line1.set_color(colormap)

        line2.set_offsets(model.end_point)

        line3.set_offsets(model.grid.agents.get('center'))

        line4.set_offsets(np.mean(model.grid.agents.get('pos'), axis=0))

        fig.canvas.draw()
        #plt.pause(0.01)'''
        i = i + 1

        model.step()
        colormap = [palette[cl] for cl in model.clusters]

        next_frame = False

    #end_time = t.time()
    #all_time = end_time - start_time
    #print(f"{all_time:.6f}s")

plt.ioff()
plt.show()