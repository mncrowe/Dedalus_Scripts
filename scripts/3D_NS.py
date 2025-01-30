'''
Solves the 3D Boussinesq Equations:

Ro*Du/Dt + k x u = -grad(p) + E*lap(u) + b*k
Ro*Db/Dt + Bu*w  = E/Pr*lap(b)

BCs: d_z(u,v,b) = 0 on z = -0.5,0.5, w = 0 on z = -0.5,0.5

'''

# Load modules:

import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
import numpy.random as random
from mpi4py import MPI

process = MPI.COMM_WORLD.rank
random.seed(process)

# Define parameters:

Nx, Ny, Nz = 128, 128, 16  # 512
Lx, Ly, Lz = 10, 10, 1

Ro = 1
Bu = 0
Pr = 1
E = 1*((Lx/Nx)**2+(Ly/Ny)**2)	  # 0.2*
Q = 1

timestep_0 = 0.005  # 0.001
t_end = 500
saves = 100  # 1000
savename = 'snapshots_128_2'  # 512
dealias = 3/2
dtype = np.float64
timestepper = d3.RK222
mesh = (4,4)
save_scale = 1  # 2
diagnostics_all = 2  # 1

# Build grids:

coords = d3.CartesianCoordinates('x', 'y', 'z')
dist = d3.Distributor(coords, dtype=dtype, mesh=mesh)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(-Lx/2, Lx/2), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(-Ly/2, Ly/2), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(-Lz/2, Lz/2), dealias=dealias)

# Define fields:

p = dist.Field(name='p', bases=(xbasis,ybasis,zbasis))
b = dist.Field(name='b', bases=(xbasis,ybasis,zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,ybasis,zbasis))
Fu = dist.VectorField(coords, name='Fu', bases=(xbasis,ybasis,zbasis))
Fb = dist.Field(name='Fb', bases=(xbasis,ybasis,zbasis))

tau_p = dist.Field(name='tau_p')
tau_b1 = dist.Field(name='tau_b1', bases=(xbasis,ybasis))
tau_b2 = dist.Field(name='tau_b2', bases=(xbasis,ybasis))
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=(xbasis,ybasis))
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=(xbasis,ybasis))

# Define equations:

x, y, z = dist.local_grids(xbasis, ybasis, zbasis)
ex, ey, ez = coords.unit_vector_fields(dist)
lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
zcross = lambda A: d3.cross(ez,A)
zdot = lambda A: d3.dot(ez,A)
grad_u = d3.grad(u) + ez*lift(tau_u1) # First-order reduction
grad_b = d3.grad(b) + ez*lift(tau_b1) # First-order reduction
KE = d3.integ(u@u/2,'z')
PE = d3.integ(b**2/2,'z')
KE_diss = E*d3.integ(d3.trace(d3.grad(u)@d3.transpose(d3.grad(u))),'z')
PE_diss = E/Pr*d3.integ(d3.grad(b)@d3.grad(b),'z')
w = d3.curl(u)
Ertel = d3.grad(b)@(w+ez)
w_rms = (d3.integ((u@ez)**2,'z'))
Surf = lambda A: A(z=Lz/2)
dx = lambda A: d3.Differentiate(A,coords[0])
dy = lambda A: d3.Differentiate(A,coords[1])
dz = lambda A: d3.Differentiate(A,coords[2])

problem = d3.IVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2], namespace=locals())
problem.add_equation("Ro*dt(u) -  E*div(grad_u) + grad(p) - b*ez + zcross(u) + lift(tau_u2) = - Ro*u@grad(u) + Fu")
problem.add_equation("Ro*dt(b) - E/Pr*div(grad_b) + Bu*zdot(u)               + lift(tau_b2) = - Ro*u@grad(b) + Fb")
problem.add_equation("trace(grad_u) + tau_p = 0")

problem.add_equation("ez@grad(b)(z=-Lz/2) = Q")
problem.add_equation("ez@grad(b)(z=Lz/2) = Q")
problem.add_equation("ez@u(z=-Lz/2) = 0")
problem.add_equation("ez@u(z=Lz/2) = 0")
problem.add_equation("ez@grad(ex@u)(z=-Lz/2) = 0")
problem.add_equation("ez@grad(ex@u)(z=Lz/2) = 0")
problem.add_equation("ez@grad(ey@u)(z=-Lz/2) = 0")
problem.add_equation("ez@grad(ey@u)(z=Lz/2) = 0")
problem.add_equation("integ(p) = 0")

# Build solver and set stop time:

solver = problem.build_solver(timestepper)
solver.stop_sim_time = t_end*(1+1/(5*saves))

# Set IC:

p0 = dist.Field(name='p', bases=(xbasis,ybasis,zbasis))
k, l, m = dist.local_modes(xbasis),dist.local_modes(ybasis),dist.local_modes(zbasis)

K_max = np.sqrt((Nx**2+Ny**2))
A1, A2 = 4, 4
K_IC1, K_IC2 = 0.1, 32        # range of K to include in initial condition
iK, iL = np.size(k,0), np.size(l,1)
cond = (K_IC2**2-k**2-l**2)*(k**2+l**2-K_IC1)

p0['c'][:,:,0] = A1/(Nx*Ny)**(1/2)*np.squeeze((2*random.rand(iK,iL,1)-1)*np.heaviside(cond,0))
p0['c'][:,:,1] = A2/(Nx*Ny)**(1/2)*np.squeeze((2*random.rand(iK,iL,1)-1)*np.heaviside(cond,0))
p0['g'] += Q/2*(z**2-1/12) # Lz/np.pi*(2/np.pi-np.cos(np.pi*z/Lz))
p0.change_scales(dealias)

u.change_scales(dealias)
p.change_scales(dealias)
b.change_scales(dealias)

p['g'] = p0['g']                    # p
b['g'] = dz(p).evaluate()['g']      # b
u['g'][0] = -dy(p).evaluate()['g']  # u
u['g'][1] = dx(p).evaluate()['g']   # v
u['g'][2] = 0                       # w

# set forcing parameters

A_Fu, A_Fb = 30, 30
K_Fu1, K_Fu2 = 8, 12
K_Fb1, K_Fb2 = 8, 12
cond_Fu = (K_Fu2**2-k**2-l**2)*(k**2+l**2-K_Fu1)
cond_Fb = (K_Fb2**2-k**2-l**2)*(k**2+l**2-K_Fb1)

# define save data

snapshots = solver.evaluator.add_file_handler(savename, sim_dt=t_end/saves)

if diagnostics_all == 1:
  snapshots.add_task(KE, name='KE', scales=save_scale)
  snapshots.add_task(PE, name='PE', scales=save_scale)
  snapshots.add_task(KE, name='KE_diss', scales=save_scale)
  snapshots.add_task(PE, name='PE_diss', scales=save_scale)
  snapshots.add_task(Surf(ez@w), name = 'vor_s', scales=save_scale)
  snapshots.add_task(Surf(p), name = 'p_s', scales=save_scale)
  snapshots.add_task(Surf(b), name = 'b_s', scales=save_scale)
  snapshots.add_task(d3.integ(u@ez,'z'), name = 'w_int', scales=save_scale)
  snapshots.add_task(w_rms, name = 'w_rms', scales=save_scale)

if diagnostics_all == 0:
  snapshots.add_task(KE+PE, name='E', scales=save_scale)
  snapshots.add_task(KE_diss+PE_diss, name='E_diss', scales=save_scale)
  snapshots.add_task(Surf(p), name = 'p_s', scales=save_scale)
  snapshots.add_task(Surf(ez@w), name = 'vor_s', scales=save_scale)
  snapshots.add_task(w_rms, name = 'w_rms', scales=save_scale)

if diagnostics_all == 2:
  snapshots.add_task(u@ex, name='u', scales=1)
  snapshots.add_task(u@ey, name='v', scales=1)
  snapshots.add_task(u@ez, name='w', scales=1)
  snapshots.add_task(b, name='b', scales=1)
  snapshots.add_task(p, name='p', scales=1)
  snapshots.add_task(w@ez, name='vor', scales=1)

#snapshots.add_task(KE, name='KE', scales=save_scale)
#snapshots.add_task(PE, name='PE', scales=save_scale)
#snapshots.add_task(KE, name='KE_diss', scales=save_scale)
#snapshots.add_task(PE, name='PE_diss', scales=save_scale)

#snapshots.add_task(Surf(ex@u), name = 'u_s', scales=save_scale)
#snapshots.add_task(Surf(ey@u), name = 'v_s', scales=save_scale)
#snapshots.add_task(Surf(b), name = 'b_s', scales=save_scale)
#snapshots.add_task(Surf(Ertel), name = 'PV_s', scales=save_scale)
#snapshots.add_task(Surf(Fb), name = 'Fb', scales=save_scale)
#snapshots.add_task(Surf(ex@Fu), name = 'Fu_x', scales=save_scale)
#snapshots.add_task(Surf(ey@Fu), name = 'Fu_y', scales=save_scale)

#snapshots.add_task(Surf(p),name='test',scales=1,layout='c')

# set CFL condition and define diagnostics

CFL = d3.CFL(solver, initial_dt=timestep_0, cadence=10, safety=0.5, threshold=0.05, max_change=1.5, min_change=0.5)
CFL.add_velocity(u)

flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(d3.integ(u@u/2+b**2/2), name='Ener')

# run simulation

while solver.proceed:
  timestep = CFL.compute_timestep()
  Fu['c'][0,:,:,0] = A_Fu/(Nx*Ny)**(1/2)*np.squeeze((2*random.rand(iK,iL,1)-1)*np.heaviside(cond_Fu,0))
  Fu['c'][1,:,:,0] = A_Fu/(Nx*Ny)**(1/2)*np.squeeze((2*random.rand(iK,iL,1)-1)*np.heaviside(cond_Fu,0))
  Fb['c'][:,:,1] = A_Fb/(Nx*Ny)**(1/2)*np.squeeze((2*random.rand(iK,iL,1)-1)*np.heaviside(cond_Fb,0))
  solver.step(timestep)
  Ener = flow.max('Ener')
  if (solver.iteration-1) % 10 == 0:
    logger.info('i=%i, t=%e, dt=%e, KE+PE=%f' %(solver.iteration, solver.sim_time, timestep, Ener))
