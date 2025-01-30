'''
Solves the SQG equation on a doubly periodic domain. The equation is

db/dt + J(psi, b) = Diss[b]

where Diss[b] represents dissipative terms such as diffusion or hyperdiffusion. In the
SQG model, the buoyancy and streamfunction are related in Fourier space by:

b = |k|tanh(H|k|) * psi

where |k| is the magnitude of the wavevector, k, and H is the layer depth. Formally, the
operator relating these two fields in a Dirichlet-to-Neumann operator and arises due to
psi satisfying the Laplace equation in the lower fluid layer due to zero PV. For H = Inf
the result simpifies to the usual SQG case:

b = |k| * psi

Here, we use a buoyancy formulation where we solve for b and calculate psi by inverting
the Fourier space relations above when evaluating the nonlinear Jacobian terms.

Matthew N. Crowe

'''

import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
from dedalus.core.domain import Domain

# define parameters...

savename = 'snapshots'
Nx, Ny = 1024, 1024
Lx, Ly = 10, 10

H = np.inf                                # layer depth, can be infinite (np.inf)

nu2 = 0*((Lx/Nx)**2+(Ly/Ny)**2)	          # viscosity
nu4 = 0.05*((Lx/Nx)**2+(Ly/Ny)**2)**2	    # hyperdiffusivity

t_end = 50                                # simulation stop time
t_saves = 100                             # number of saves
timestep = 0.01                           # initial timestep if CFL_on = True

CFL_on = True                             # 1 - use variable timestep, 0 - use fixed dt
save_u = False                            # if True, saves velocity fields (u,v)
t_save = t_end/t_saves                    # time between saves
dealias = 3/2                             # 3/2 required for quadratic nonlinearities
timestepper = d3.RKSMR                    # options: RK111, RK222, RKSMR, RK443
dtype = np.float64

# create domain and fields...

coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(-Lx/2, Lx/2), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(-Ly/2, Ly/2), dealias=dealias)
domain = Domain(dist, (xbasis, ybasis))

b = dist.Field(name='b', bases=(xbasis,ybasis))

x, y = dist.local_grids(xbasis, ybasis)
k, l = dist.local_modes(xbasis), dist.local_modes(ybasis)

# define GeneralFunction for b -> psi inversion step...

K2 = 4*np.pi**2*(np.floor(k/2)**2/Lx**2 + np.floor(l/2)**2/Ly**2)  # convert local modes to wavenumbers and calculate |k|^2
eps = np.finfo(dtype).eps                                          # small eps included in DN to avoid div-by-0 warnings
DN = 1/(np.sqrt(eps + K2) * np.tanh(H * np.sqrt(eps + K2)))        # define inversion operator in Fourier space
DN[K2==0] = 0

def _b_to_psi(*args):
    return args[0]['c'] * DN

def b_to_psi(*args, domain=domain, F=_b_to_psi):
    return d3.GeneralFunction(dist=dist, domain=domain, tensorsig=(), dtype=np.float64, layout='c', func=F, args=args)

# create IVP...

problem = d3.IVP([b], namespace=locals())

hyp = lambda A: d3.lap(d3.lap(A))
dx = lambda A: d3.Differentiate(A,coords[0])
dy = lambda A: d3.Differentiate(A,coords[1])
J = lambda A,B: dx(A)*dy(B)-dx(B)*dy(A)
psi = b_to_psi(b)

problem.add_equation("dt(b) + nu4*hyp(b) - nu2*lap(b) = - J(psi, b)")

solver = problem.build_solver(timestepper)
solver.stop_sim_time = t_end

# set IC...

b['g'] = np.exp(-(x**2+2*y**2))                               # set psi
b['c'] = b['c'] * np.sqrt(K2)*np.tanh(H * np.sqrt(eps + K2))  # convert to b using multiplication by |k|tanh(K|k|)

# save output...

snapshots = solver.evaluator.add_file_handler(savename, sim_dt=t_save, max_writes=100)
snapshots.add_task(psi, name='psi', scales=1)
snapshots.add_task(b, name='b', scales=1)
snapshots.add_task(d3.integ(b**2), name='B', scales=1)        # bouyancy variance
snapshots.add_task(d3.integ(psi*b/2), name='E', scales=1)     # domain integrated energy
if save_u:
  snapshots.add_task(-dy(psi), scales=1, name='u')
  snapshots.add_task(dx(psi), scales=1, name='v')

# CFL condition...

if CFL_on:
  CFL = d3.CFL(solver, initial_dt=timestep, cadence=5, safety=0.5, threshold=0.1,max_change=1.5, min_change=0.5, max_dt=t_save/5)
  CFL.add_velocity(d3.skew(d3.grad(psi)))

# flow properties...

flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(d3.integ(b**2), name='BVar')
flow.add_property(d3.integ(psi*b/2), name='Energy')

# main loop...

while solver.proceed:
  if CFL_on:
    timestep = CFL.compute_timestep()
  solver.step(timestep)
  if (solver.iteration-1) % 100 == 0:
    Ener = flow.max('Energy')
    BVar = flow.max('BVar')
    logger.info('Iteration=%i, Time=%e, dt=%e, E=%f, B=%f' %(solver.iteration, solver.sim_time, timestep, Ener, BVar))
