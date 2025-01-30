# solves the 2D QG equations on a doubly periodic domain

import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
import sys
import numpy.random as random

# load arguments...

if len(sys.argv)>1:
  savename = str(sys.argv[1])
else:
  savename = 'snapshots' # name of data files and folders

if len(sys.argv)>2:
  Nx = int(sys.argv[2])
else:
  Nx = 1024    # x gridpoints

if len(sys.argv)>3:
  Ny = int(sys.argv[3])
else:
  Ny = 1024    # y gridpoints

if len(sys.argv)>4:
  Lx = int(sys.argv[4])
else:
  Lx = 10    # x domain length

if len(sys.argv)>5:
  Ly = int(sys.argv[5])
else:
  Ly = 10    # y domain length

if len(sys.argv)>6:
  beta = int(sys.argv[6])
else:
  beta = 0.0    # beta parameter

if len(sys.argv)>7:
  R = int(sys.argv[7])
else:
  R = np.inf    # barotropic Rossby radius

if len(sys.argv)>8:
  nu2 = float(sys.argv[8])*((Lx/Nx)**2+(Ly/Ny)**2)
else:
  nu2 = 0*((Lx/Nx)**2+(Ly/Ny)**2)	  # viscosity

if len(sys.argv)>9:
  nu4 = float(sys.argv[9])*((Lx/Nx)**2+(Ly/Ny)**2)**2
else:
  nu4 = 1*((Lx/Nx)**2+(Ly/Ny)**2)**2	  # hyperdiffusion

if len(sys.argv)>10:
  t_end = float(sys.argv[10])
else:
  t_end = 100  # stop time

if len(sys.argv)>11:
  t_saves = int(sys.argv[11])
else:
  t_saves = 100    # time saves, normally 250 is max for 2048^2 domain

if len(sys.argv)>12:
  timestep = float(sys.argv[12])
else:
  timestep = 1e-6	# initial timestep, usually 0.001 for high resolution (2048^2)

if len(sys.argv)>13:
  nu0 = float(sys.argv[12])
else:
  nu0 = 2e-3	# large scale damping to remove energy at largest scales

# other parameters...

CFL_on = 1                    # 1 - use variable timestep, 0 - use fixed dt
t_save = t_end/t_saves        # time between saves
t_end = t_end+t_save/5        # reset t_end such that final save is always completed
dealias = 3/2
timestepper = d3.RK222
dtype = np.float64

# stochastic forcing ...

F_A = 1e1                     # forcing amplitude, full amplitude is F_A/sqrt(Nx*Ny*dt)
K_F1, K_F2 = Nx/8-4, Nx/8+4   # forcing wavenumber bounds, wavenumbers in [K_F1, K_F2] are forced randomly
max_dt = 1e-1                 # max dt set by forcing

# create domain and fields...

coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(-Lx/2, Lx/2), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(-Ly/2, Ly/2), dealias=dealias)

psi = dist.Field(name='psi', bases=(xbasis,ybasis))
F = dist.Field(name='F', bases=(xbasis,ybasis))
tau = dist.Field(name='tau')
x, y = dist.local_grids(xbasis, ybasis)
k, l = dist.local_modes(xbasis),dist.local_modes(ybasis)

# create IVP...

problem = d3.IVP([psi,tau], namespace=locals())

hyp = lambda A: d3.lap(d3.lap(A))             # del^4 operator
dx = lambda A: d3.Differentiate(A,coords[0])
dy = lambda A: d3.Differentiate(A,coords[1])
J = lambda A,B: dx(A)*dy(B)-dx(B)*dy(A)

Q = d3.lap(psi)-1/R**2*psi                    # Potential vorticity anomaly
E = d3.grad(psi)@d3.grad(psi)/2+psi**2/R**2   # Energy, equiv. -psi*Q/2
Z = Q**2/2                                    # Enstrophy
I = d3.integ(F*psi)                           # Energy injection rate

problem.add_equation("dt(Q) + beta*dx(psi) + nu4*hyp(Q) - nu2*lap(Q) - nu0*psi + tau = - F - J(psi,Q)")
problem.add_equation("integ(psi) = 0")

solver = problem.build_solver(timestepper)
solver.stop_sim_time = t_end

# set IC...

K_max = np.sqrt((Nx**2+Ny**2))
A_psi = 0
K_IC1, K_IC2 = 0.1, K_max/8        # range of K to include in initial condition
iK = np.size(k,0)
iL = np.size(l,1)

psi['g'] = np.exp(-((x+2*Lx/3)**2+(y+2*Lx/3)**2)/(5*Lx)**2) - np.exp(-((x-2*Lx/3)**2+(y-2*Lx/3)**2)/(5*Lx)**2)
psi['c'] += A_psi/(Nx*Ny)*(2*random.rand(iK,iL)-1)*(1e-12+np.sqrt(k**2+l**2)/K_max)**(-11/6)*np.heaviside((K_IC2**2-k**2-l**2)*(k**2+l**2-K_IC1),0)

# save output...

snapshots_h = solver.evaluator.add_file_handler(savename, sim_dt=t_save, max_writes=100)
snapshots_h.add_task(psi, name='psi', scales=1)
snapshots_h.add_task(Q, name='Q', scales=1)
#snapshots_h.add_task(-dy(psi), name='u', scales=sh)
#snapshots_h.add_task(dx(psi), name='v', scales=sh)
snapshots_h.add_task(d3.integ(Z), name='Z', scales=1)
snapshots_h.add_task(d3.integ(E), name='E', scales=1)
snapshots_h.add_task(I, name='I', scales=1)

# CFL condition ...

if CFL_on == 1:
  CFL = d3.CFL(solver, initial_dt=timestep, cadence=10, safety=0.5, threshold=0.1,max_change=1.5, min_change=0.5, max_dt=min(max_dt,t_save/5))
  CFL.add_velocity(d3.skew(d3.grad(psi)))

# flow properties...

flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(d3.integ(E), name='Energy')
flow.add_property(d3.integ(Z), name='Enstrophy')

# main loop...

try:
    logger.info('Starting main loop')
    while solver.proceed:
        if CFL_on == 1:
          timestep = CFL.compute_timestep()
          F['c'] = F_A/np.sqrt(Nx*Ny*timestep)*(2*random.rand(iK,iL)-1)*np.heaviside((K_F2**2-k**2-l**2)*(k**2+l**2-K_F1),0)
          solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            Ener = flow.max('Energy')
            Enst = flow.max('Enstrophy')
            logger.info('Iteration=%i, Time=%e, dt=%e, E=%f, Z=%f' %(solver.iteration, solver.sim_time, timestep, Ener, Enst))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
