"""
Two-dimensional non-Cartesian example: QG dynamics on a sphere

Du/Dt + 2*Omega*k x u = -\grad(p) + nu*lap^2(u)
div(u) = 0

for Rotation vector Omega*k and (hyper) viscosity nu.

Solves the two-dimensional rotating Euler system on a sphere. This is
a simple model for Rossby waves in the Earth's atmosphere.

"""

import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
import numpy.random as random

R = 6.4e6
Omega = 7.3e-5
nu = 1e3

Ntheta,Nphi = 128,256
timestep = 1e3
t_end = 1e7
saves = 100
savename = 'QG_Sphere_Example'
timestepper = d3.RK222
dealias = 3/2
dtype = np.float64

coords = d3.S2Coordinates('phi', 'theta')
dist = d3.Distributor(coords, dtype=dtype)
basis = d3.SphereBasis(coords, (Nphi, Ntheta), radius=R, dealias=dealias, dtype=dtype)

u = dist.VectorField(coords, name='u', bases=basis)
p = dist.Field(name='p', bases=basis)
tau = dist.Field(name='tau')

phi, theta = dist.local_grids(basis)
zcross = lambda A: d3.MulCosine(d3.skew(A))

problem = d3.IVP([u, p, tau], namespace=locals())
problem.add_equation("dt(u) + nu*lap(lap(u)) + grad(p) + 2*Omega*zcross(u) = - u@grad(u)")
problem.add_equation("div(u) + tau = 0")
problem.add_equation("Average(p) = 0")

solver = problem.build_solver(d3.RK222)
solver.stop_sim_time = t_end

snapshots = solver.evaluator.add_file_handler(savename, sim_dt=t_end/saves)
snapshots.add_task(p, name='p')
snapshots.add_task(-d3.div(d3.skew(u)), name='vor')

u['g'][0] = 100*np.sin(3*theta)
u['g'] += 10*random.rand(2,np.size(Nphi),np.size(Ntheta))

CFL = d3.CFL(solver, initial_dt=timestep, cadence=10, safety=0.5, threshold=0.05, max_change=1.5, min_change=0.5)
CFL.add_velocity(u)

while solver.proceed:
  timestep = CFL.compute_timestep()
  solver.step(timestep)
  if (solver.iteration-1) % 10 == 0:
    logger.info('i=%i, t=%e, dt=%e' %(solver.iteration, solver.sim_time, timestep))
