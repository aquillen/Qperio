

# routines for classical system integration of perturbed Harper model

# updated so consistent with Qperio11.ipynb


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# set up equations of motion
# y = [phi,p]  dimension 2
# H(phi,p,tau) = a(1 - cos (p-b)) - epsilon*cos phi - mu*cos(phi - nu t) - mup*cos(phi + tau)
# dy/dt = [\dot phi, \dot p]
#       = [a\sin p, -epsilon*sin(phi) - mu*sin(phi - nu*t) - mup*sin(phi + tau) ]
#There should be three islands, one at p=0 and the others at sin p=1/a
#the widths of these are set by \sqrt{2 epsilon/a}, \sqrt{2 mu/a},  \sqrt{2 mup/a}

# dy/dtau = func(y,tau), needed for integrator!
# note that y[0] = phi and y[1] = p
def cfunc(y,tau,a,b,epsilon,mu,mup):
    return [a*np.sin(y[1]-b),\
            -epsilon*np.sin(y[0]) - mu*np.sin(y[0] - tau) - mup*np.sin(y[0] + tau)]

# this routine is used to calculate energy value
def H_classical(p,phi,tau,a,b,epsilon,mu,mup):
    #phi = y[0]; p = y[1]
    return a*(1.0 - np.cos(p-b)) - epsilon*np.cos(phi)\
        - mu*np.cos(phi-tau) - mup*np.cos(phi+tau)
        
# for unperturbed energy values, classical system
def H0_classical(p,phi,a,b,epsilon):
        return a*(1.0 - np.cos(p-b)) - epsilon*np.cos(phi)
        
        
# integrate at a period, the forced pendulum 
# arguments:
#   y0: is initial conditions, 
#   npoints: is number of points we want
#   a,epsilon, mu, mup: are parameters of the Hamiltonian model
#   taushift:  points of output are at times  taushift + 2 pi n, integer n up to npoints 
#
#   integrate every 2 pi period, npoints returned including initial condition 
# returns: phi,p arrays of integrated points 
# this particular routine does all times at once by sending a set of times to the integrator
# if we want to take p modulo 2pi we would have to integrate slowly to find when trajectories
# cross pi or -pi
twopi = 2.0*np.pi
def givepoints_arr(y0,npoints,a,b,epsilon,mu,mup,taushift):
    # set up time array, every 2pi/nu so for period of perturbation
    step  = 2.0*np.pi
    stop = step*npoints
    time  = np.arange(0.0,stop,step)  # time array for outputs  goes from 0 to stop-step 
    #                              with increment step, does not reach stop. 
    time += taushift  # shift time vector 
    # do the integration
    y = odeint(cfunc, y0, time, args=(a,b,epsilon,mu,mup))
    #y2 = odeint(func, y0, time, Dfun=jacobian, args=(epsilon,mu,nu))  # if you want to use a jacobian
    phi_arr = np.squeeze(y[:,0])  # is an array of phi at different times
    p_arr = np.squeeze(y[:,1])    # is an array of momenta at different times
    
    phi_arr  = phi_arr%twopi  # so that phi in [0:2pi]
    p_arr = p_arr%twopi  # so that p is in [0,2pi]
    
    ii = phi_arr >np.pi
    phi_arr[ii]-= twopi # now phi is in [-pi,pi]
    ii = p_arr >np.pi
    p_arr[ii]-= twopi   # now p is in [-pi,pi]
    
    return phi_arr,p_arr 
       
       
# integrate points and plot them on axis   ax
def givepoints(y0,npoints,a,b,epsilon,mu,mup,marker,ax):
    phi_arr,p_arr=givepoints_arr(y0,npoints,a,b,epsilon,mu,mup)
    ax.plot(phi_arr,p_arr,marker,markersize=0.5) # plot it
    
colorlist = ['black','blue','green','magenta','red','orange','gold','blueviolet','springgreen','dodgerblue']


# randomly choose an initial condition, integrate and plot in axis ax
# arguments:
#    npoints: numbers of points to plot
#    pmin, pmax constrain the vertical width of randomly chosen initial conditions
#    a,epsilon,mu,mup,b: parameters for the dynamical model
#    ax  the axis on which to plot points
def rand_give_points(npoints,a,b,epsilon,mu,mup,taushift,pmin,pmax,ax):
    phi = np.random.uniform(low=-np.pi,high=np.pi)
    p = np.random.uniform(low=pmin,high=pmax)
    y0 = [phi,p]
    phi_arr,p_arr=givepoints_arr(y0,npoints,a,b,epsilon,mu,mup,taushift)
    i = np.random.randint(low=0,high =len(colorlist))
    ax.scatter(phi_arr,p_arr,s=1,edgecolor='none',facecolor=colorlist[i],lw=1)



# make a surface of section figure with orbits
# inputs: a classical model class, and a label
def mkfig_cl(cla,alabel):
    norb = cla.norb
    npoints = cla.npoints
    a = cla.a
    b = cla.b
    eps = cla.eps
    mu = cla.mu
    mup = cla.mup
    taushift = cla.taushift
    froot = cla.froot
    
    # set up display
    fig,ax = plt.subplots(1,1,figsize=(3,3),dpi=200)
    plt.subplots_adjust(bottom=0.18,top=0.90,left=0.18,right=0.98)
    ax.text(2.6,-3.9,r'$\phi$',fontsize=14)
    ax.set_ylabel('p',labelpad=0)
    ax.set_aspect('equal')

    fac=1.01
    xmax = np.pi;  ymax = np.pi;  ymin = -ymax
    ax.set_xlim([-xmax*fac,xmax*fac])
    ax.set_ylim([ ymin*fac,ymax*fac])

    parm_label = r'$a$={:.2f},'.format(a)
    parm_label += r'$\epsilon$={:.2f},'.format(eps)
    parm_label += r'$\mu$={:.2f},'.format(mu)
    muprime = r'$\mu$' + '\''
    parm_label += muprime
    parm_label += r'={:.2f},'.format(mup)
    #parm_label += r'$\nu$={:.0f}'.format(nu)
    ax.text(0,3.4,parm_label,ha='center',va='center')

    for i in range(norb):  # this many orbits
        rand_give_points(npoints,a,b,eps,mu,mup,taushift,ymin,ymax,ax)
        
    if (len(alabel)>1):
        plt.text(-4.5,3.2,alabel,fontsize=14)
    if (len(froot)>2):
        ofile = froot + '_class.png'
        plt.savefig(ofile,dpi=300)
        
    plt.show()



# a class to store classical model info
#    norb:  numbers of orbits
#    npoints: numbers of points to plot
#    a,epsilon,mu,mup,taushfit,b: parameters for the dynamical model
#    froot: a root for labeling files
class Hcla:
    def __init__(self,norb,npoints,a,b,eps,mu,mup,taushift,froot):
        self.a = a
        self.b = b
        self.eps = eps
        self.mu = mu
        self.mup = mup
        self.taushift = taushift
        self.norb = norb
        self.npoints = npoints
        self.froot = froot
        
    # randomly choose an norb initial conditions, integrate and compute <h_0> and sqrt(<h_0^2 - <h_0>^2>)
    #  from the orbits
    # compute   mu_arr: mean H0 of each orbit
    # compute   sig_arr: standard deviation of H0 for each orbit
    # store this info in the class itself
    def sigh0_orbs(self):
        mu_arr = np.zeros(self.norb)
        sig_arr = np.zeros(self.norb)
        for i in range(self.norb):
            phi = np.random.uniform(low=-np.pi,high=np.pi)
            p = np.random.uniform(low=-np.pi,high=np.pi)
            y0 = [phi,p]
            phi_arr,p_arr=givepoints_arr(y0,self.npoints,self.a,self.b,self.eps,self.mu,self.mup,self.taushift)
            Elist = H0_classical(p_arr,phi_arr,self.a,self.b,self.eps)
            mu_arr[i] = np.mean(Elist)
            sig_arr[i] = np.std(Elist)
            
        iphi = np.argsort(mu_arr) # in order of increasing energy
        self.mu_arr = mu_arr[iphi]
        self.sig_arr = sig_arr[iphi]
        #return mu_arr,sig_arr
    
    def print_info(self):
        # compute resonance half widths
        wid0 = 2*np.arcsin(np.sqrt(np.abs(self.eps/self.a)))
        widmu = 2*np.arcsin(np.sqrt(np.abs(self.mu/self.a)))
        widmup = 2*np.arcsin(np.sqrt(np.abs(self.mup/self.a)))
        print('res widths {:.2f}, {:.2f}, {:.2f}'.format(wid0,widmu,widmup))
    
        if (self.a<1):
            print('res locs p=0,+-none, a <1')
        else:
            p_mu = np.arcsin(1/self.a)  # a resonance location
            print('res locs p=0,+-{:.2f}'.format(p_mu))
            overlap1 = p_mu - (wid0+widmu)
            overlap2 = p_mu - (wid0+widmup)
            print('overlaps {:.2f} {:.2f}'.format(overlap1,overlap2)) # guess of overlaps
        
        omega0 = np.sqrt(self.a*self.eps) # frequency of primary resonance
        print('omega0 = {:.2f}'.format(omega0))
        
        # not sure what this is right now
        deltH = 4*np.pi*self.mu/omega0  * np.exp(-0.5*np.pi/omega0)  # this is the half width about
        print('deltH = {:.3f}'.format(deltH))
        #range of energy is 2*(eps+a), range of p is 2pi so we guess at
        Deltp = 2*deltH/(2*self.eps + 2*self.a) *2*np.pi # change in p is
        print('Deltp = {:.3f}'.format(Deltp))
