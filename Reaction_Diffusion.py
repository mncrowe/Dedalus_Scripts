"""
Two-dimensional real coupled PDE example: Reaction-Diffusion

du/dt = Du*lap(u) + u^2*v - (a+b)*u
dv/dt = Dv*lap(v) - u^2*v + a*(1-v)

for parameters a, b, D.

Solves a reaction-diffusion model (Gray-Scott) with 3 variable parameters.
Usually we take D = 2 and vary (a,b). A wide range of interesting behaviours
arise from different parameter values.

"""

import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

a,b,Du,Dv = 0.037,0.063,1e-4,2e-4
Nx,Ny = 256,256
Lx,Ly = 10,10

timestep = 0.5
t_end = 5e4
saves = 50
savename = 'Reaction_Diffusion_Example'
timestepper = d3.RK222
dealias = 1
dtype = np.float64

coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(-Lx/2, Lx/2),dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(-Ly/2, Ly/2),dealias=dealias)

u = dist.Field(name='u', bases=(xbasis,ybasis))
v = dist.Field(name='v', bases=(xbasis,ybasis))
x, y = dist.local_grids(xbasis, ybasis)

problem = d3.IVP([u,v], namespace=locals())

problem.add_equation("dt(u) - Du*lap(u) + (a+b)*u = u**2*v")
problem.add_equation("dt(v) - Dv*lap(v) + a*v = a - u**2*v")

solver = problem.build_solver(timestepper)
solver.stop_sim_time = t_end

u['g'] = 1*np.exp(-(x**2+y**2)/0.1**2)
v['g'] = 1

snapshots = solver.evaluator.add_file_handler(savename, sim_dt=t_end/saves)
snapshots.add_task(u, name='u', scales=2)
snapshots.add_task(v, name='v', scales=2)

while solver.proceed:
  solver.step(timestep)
  if (solver.iteration-1) % 10 == 0:
    logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration, solver.sim_time, timestep))
