""""
Two-dimensional complex PDE example: Schrodinger equation

[i*d/dt + del^2 ] psi = 0

for complex unit i and wavefunction psi

Solves the Schrodinger equation for a free particle (potential V = 0), All
parameters, Plank's constant and mass, have been scaled out. This example
demonstrates that Dedalus works with complex values fields too. We impose
walls at the coordinate boundaries where psi = 0, this is enforces using 
an odd symmetry in the Fourier basis.

"""

import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

Nx,Ny = 256,256
Lx,Ly = 10,10

timestep = 0.01
t_end = 1
saves = 100
savename = 'Schrodinger_Example'
timestepper = d3.RK222
dtype = np.complex128

coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.ComplexFourier(coords['x'], size=Nx, bounds=(-Lx/2, Lx/2))
ybasis = d3.ComplexFourier(coords['y'], size=Ny, bounds=(-Ly/2, Ly/2))

psi = dist.Field(name='psi', bases=(xbasis,ybasis))
x, y = dist.local_grids(xbasis, ybasis)

problem = d3.IVP([psi], namespace=locals())

problem.add_equation("1j*dt(psi) + lap(psi) = 0")

solver = problem.build_solver(timestepper)
solver.stop_sim_time = t_end

psi['g'] = (1+1j)*np.exp(-10*((x-2)**2+y**2))
#psi['c'][0::2,0::2] = 0

snapshots = solver.evaluator.add_file_handler(savename, sim_dt=t_end/saves)
snapshots.add_task(psi, name='psi', scales=1)

while solver.proceed:
  solver.step(timestep)
  if (solver.iteration-1) % 10 == 0:
    logger.info('Iteration=%i, Time=%e, dt=%e' %(solver.iteration, solver.sim_time, timestep))
