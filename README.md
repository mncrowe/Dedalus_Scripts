# Dedalus Scripts

A repository for various Dedalus scripts. Included are:

- ODE.py: a silly example showing you can use a PDE solver to solve a simple coupled ODE system
- QG_sphere: solves the 2D rotating Euler equations on a sphere, modified from the [shallow water](https://dedalus-project.readthedocs.io/en/latest/pages/examples/ivp_sphere_shallow_water.html) example
- Reaction_Diffusion.py: solves the Gray-Scott model for two reacting (and diffusing) substances
- Schrodinger.py: solves the Schrodinger equation for a free particle using complex fields
- Kelvin_Helmholtz.py: solves the 2D Boussinesq equations with a stratified shear layer as the initial condition
- 2D_QG.py: solves the single layer (2D) quasi-geostrophic equation with a random forcing term
- 3D_NS.py: solves the 3D (rotating) Boussinesq equations with a random forcing term
- SQG.py: an implementation of the Surface Quasi-Geostrophic equation with finite depth capability

These scripts use [Dedalus3](https://dedalus-project.readthedocs.io/en/latest/index.html) and are written in Python. The first 5 scripts in the list above were written as examples for a research skills session I ran at UCL for PhD students. They demonstrate some interesting physics and various things you can do in Dedalus but are not intended for serious research. The remaining 3 scripts solve quasi-geostrophic and Boussinesq systems and variants of these have been used in my [research](https://mncrowe.github.io/publications/).
