"""
Coupled ODE example: Lokta-Volterra equations

dx/dt =  alpha x - beta  xy
dy/dt = -gamma y + delta xy

for (positive) parameters alpha, beta, gamma, delta

Dedalus isn't really designed for ODE problems but has some good timestepping
schemes we might want to make use of. We technically define a spatial coordinate
named 's' but never assign it a basis. Variables, here x and y, are then defined
as constants (i.e. s independent) and solved for by adding the ODE system to the
solver. Initial conditions, data snapshots and running proceed as usual.

"""

import numpy as np
import dedalus.public as d3

alpha,beta,gamma,delta = 2/3,4/3,1,1

timestep = 0.01
t_end = 10
saves = 100
savename = 'ODE_Example'
timestepper = d3.RK222
dtype = np.float64

coords = d3.CartesianCoordinates('s')      # define s as an unused placeholder coordinate
dist = d3.Distributor(coords, dtype=dtype)

x = dist.Field(name='x')
y = dist.Field(name='y')

problem = d3.IVP([x,y], namespace=locals())

problem.add_equation("dt(x) - alpha*x = -beta*x*y")
problem.add_equation("dt(y) + gamma*y = delta*x*y")

solver = problem.build_solver(timestepper)
solver.stop_sim_time = t_end

x['g'] = 1
y['g'] = 1

snapshots = solver.evaluator.add_file_handler(savename, sim_dt=t_end/saves)
snapshots.add_task(x, name='x')
snapshots.add_task(y, name='y')

while solver.proceed:
  solver.step(timestep)

print('Run Complete')