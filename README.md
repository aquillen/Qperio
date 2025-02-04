# Qperio
Exploring sinusoidal perturbed Harper/pendulum classical and quantum analogs.
The Hamiltonian in both classical and quantum settings is

$$ H(p,\phi,\tau) = a(1 - \cos p) - \epsilon \cos \phi - \mu \cos(\phi - \tau) - \mu' \cos(\phi + \tau)$$

In the classical setting we create surfaces of section mapping at a period of $2 \pi$. 

In the quantum setting $p, \phi$ become operators on a discrete $N$ dimensional quantum space and we compute the propagator across a period of $2 \pi$. 

Relevant for this paper:  https://arxiv.org/abs/2412.14926 on the topic of chaos excited at a separatrix. 

Qcomp01.ipynb, Qperio11.ipynb, Qergodic01.ipynb are used to make figures for this manuscript. 

Qperio11.ipynb creates the propagators and integrates the associated classical models.  Most of the figures in the manuscript are created here. 

Qergodic.ipynb makes Husimi functions near the sepatrix of an ergodic model and compares what they look like as $N$ (the dimension of the quantum system) increases.

Qcomp01.ipynb computes commutators and matrix elements to check that our derived analytical estimates for the width of the ergodic region are reasonable. 
