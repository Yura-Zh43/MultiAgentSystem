import mesa as ms
import solara as s
from Model import Model
from mesa.visualization.components import AgentPortrayalStyle

def draw(agent):
    boid_style = AgentPortrayalStyle(color="red", size=20)
    return boid_style

n_agents = 10
model = Model(n_agents)

model.step()

model_params = {
    "n_agents": 10
}

renderer = ms.visualization.SpaceRenderer(
    model,
    backend="matplotlib",
).render(agent_portrayal=draw)

page = ms.visualization.SolaraViz(
    model,
    renderer,
    model_params=model_params,
    name="Model",
)
page  # noqa