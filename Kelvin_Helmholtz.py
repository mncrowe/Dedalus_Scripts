"""
Two-dimensional vector PDE example with boundaries: Kelvin-Helmholtz

Du/dt = -grad(p) + 1/Re*lap(u) + bk
div(u) = 0
Db/Dt = Pr/Re*lap(b)

for vertical basis vector k and Reynolds number Re and Prandtl number Pr.

Solves the 2D Boussinesq equations with a stratified shear layer as the initial
condition. Kelvin-Helmholtz instabilities develop and roll up into nonlinear billows.

"""

import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

Re,Pr = 12000,1    # 3000,1
h,S0,N02 = 0.1,10,10  # stratified shear layer parameters
Nx,Nz = 2048,1024      # 512,256
Lx,Lz = 1.4,1        # 1.4,1

timestep = 0.001      # 1e-3
t_end = 20
saves = 200
savename = 'Kelvin_Helmholtz_Example'
timestepper = d3.RK222
dealias = 3/2
dtype = np.float64

coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(-Lz/2, Lz/2), dealias=dealias)

p = dist.Field(name='p', bases=(xbasis,zbasis))
b = dist.Field(name='b', bases=(xbasis,zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))

tau_p = dist.Field(name='tau_p')
tau_b1 = dist.Field(name='tau_b1', bases=xbasis)
tau_b2 = dist.Field(name='tau_b2', bases=xbasis)
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)

x, z = dist.local_grids(xbasis, zbasis)
ex, ez = coords.unit_vector_fields(dist)
lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
grad_u = d3.grad(u) + ez*lift(tau_u1) # First-order reduction
grad_b = d3.grad(b) + ez*lift(tau_b1) # First-order reduction

problem = d3.IVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2], namespace=locals())
problem.add_equation("dt(u) -  1/Re*div(grad_u) + grad(p) - b*ez + lift(tau_u2) = - u@grad(u)")
problem.add_equation("dt(b) - Pr/Re*div(grad_b) +                  lift(tau_b2) = - u@grad(b)")
problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("ez@grad(b)(z=-Lz/2) = 0")
problem.add_equation("ez@grad(b)(z=Lz/2) = 0")
problem.add_equation("ez@u(z=-Lz/2) = 0")
problem.add_equation("ez@u(z=Lz/2) = 0")
problem.add_equation("ez@grad(ex@u)(z=-Lz/2) = 0")
problem.add_equation("ez@grad(ex@u)(z=Lz/2) = 0")
problem.add_equation("integ(p) = 0")

solver = problem.build_solver(timestepper)
solver.stop_sim_time = t_end

w0 = 0*x*z
u0 = 0*x*z
for i in range(1,2):
  w0 += 1e-3*np.cos(2*i*np.pi*x/Lx-i**2)*np.cos(np.pi*z/Lz)/(1/Lz)
  u0 += 1e-3*np.sin(2*i*np.pi*x/Lx-i**2)*np.sin(np.pi*z/Lz)/(2*i/Lx)

#b.fill_random('g', seed=42, distribution='normal', scale=1e-5) # Random noise
b['g'] += N02*h*np.tanh(z/h)   # Add shear layer and small perturbation
u['g'][0] = S0*h*np.tanh(z/h)+u0
u['g'][1] = w0

snapshots = solver.evaluator.add_file_handler(savename, sim_dt=t_end/saves)
snapshots.add_task(u@ex, name='u', scales=1)
snapshots.add_task(u@ez, name='w', scales=1)
snapshots.add_task(b, name='b', scales=1)

CFL = d3.CFL(solver, initial_dt=timestep, cadence=10, safety=0.5, threshold=0.05, max_change=1.5, min_change=0.5)
CFL.add_velocity(u)

flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(d3.integ(u@u/2), name='Ener')
flow.add_property(ex@u, name='um')
flow.add_property(ez@u, name='wm')

while solver.proceed:
  timestep = CFL.compute_timestep()
  solver.step(timestep)
  Ener = flow.max('Ener')
  Umax = flow.max('um')
  Wmax = flow.max('wm')
  if (solver.iteration-1) % 10 == 0:
    logger.info('i=%i, t=%e, dt=%e, E=%f, Max(u,w)=(%f,%f)' %(solver.iteration, solver.sim_time, timestep, Ener, Umax, Wmax))