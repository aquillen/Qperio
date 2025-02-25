#
# contains routines for displacement operator, coherent states and Husimi distribution
# also for the constructing U_T propagator, and sorting eigenstates states

###################### updated to Qperio11

import numpy as np
import mpmath as mpmath
import matplotlib.pyplot as plt


# return Xhat as a 2d NxN complex matrix (unitary)
def Xhat_op(N): # in regular basis
    X = np.zeros((N,N),dtype=complex)
    for k in range(N):
        X[(k+1)%N,k] = 1.0 + 0j
    return X

# return Zhat as a 2d NxN complex matrix (unitary)
def Zhat_op(N):  # in regular basis
    Z = np.zeros((N,N),dtype=complex)
    for k in range(N):
        omegaj = np.exp(2*np.pi*k*1j/N)
        Z[k,k] = omegaj
    return Z
    
# adjoint of Zhat
def Zhat_dagger_op(N):
    return np.conjugate(Zhat_op(N))

# adjoint of Xhat
def Xhat_dagger_op(N):
    return np.transpose(Xhat_op(N))

# additional discrete ops
def cos_phi_op(N):
    return (Zhat_op(N) + Zhat_dagger_op(N))*0.5

def sin_phi_op(N):
    return (Zhat_op(N) - Zhat_dagger_op(N))/(2.0j)

def cos_p_op(N):
    return (Xhat_op(N) + Xhat_dagger_op(N))*0.5

def sin_p_op(N):
    return (Xhat_op(N) - Xhat_dagger_op(N))/(2.0j)

# create Dhat, the displacement operator as a function of k,l  
# returns an NxN complex matrix, is unitary 
def Displacement_op(N,k,l):
    #D = np.zeros((N,N),dtype=complex)
    omega = np.exp(2*np.pi*1j/N)
    #omfac = np.power(-omega*k*l/2.0)
    X = Xhat_op(N)
    Z = Zhat_op(N)
    Xl = np.linalg.matrix_power(X, l)
    Zk = np.linalg.matrix_power(Z, k)
    Dhat= np.power(omega,-k*l/2.0) * np.matmul(Zk,Xl)
    return Dhat

# compute the eta function in two different ways (and I checked that they agree!)
# returns an N dimensional complex array (a normalized quantum state vector)
def eta_tilde_check(N):
    eta = np.zeros(N,dtype=complex)
    eta_b = np.zeros(N,dtype=complex)
    efac = np.exp(-np.pi/N)
    for m in range(N):
        eta[m] = np.real(mpmath.jtheta(3,np.pi*m/N,efac)/np.sqrt(N))
    for m in range(N):  #checking with a direct sum 
        for j in range(-2*N,2*N):
            eta_b[m] += np.exp(-np.pi/N*(m + j*N)**2)  
            # arbitrarily stopped at 2N, the sum should go to infinity 
    eta_mag = np.sqrt(np.sum(eta*np.conj(eta)))
    eta = eta/eta_mag # normalize
    eta_b_mag = np.sqrt(np.sum(eta_b*np.conj(eta_b)))
    eta_b = eta_b/eta_b_mag # normalize
    return eta,eta_b
    
    
# computes eigenfunction |\tilde eta> of the QFT
# returns an N dimensional complex normalized statevector
# This is a unimodal periodic thing which is peaked at index 0
def eta_tilde(N):
    eta = np.zeros(N,dtype=complex)
    efac = np.exp(-np.pi/N)
    for m in range(N):  # compute using theta function
        eta[m] = np.real(mpmath.jtheta(3,np.pi*m/N,efac)/np.sqrt(N))
    eta_mag = np.sqrt(np.sum(eta*np.conj(eta)))
    eta = eta/eta_mag   # normalize
    return eta

#  Creat a coherent state from Displacement operator hat D and eigenfunction |tilde eta>
# returns an N dimensional complex normalized state vector
def coherent_state(N,k,l):
    Dhat = Displacement_op(N,k,l)
    eta=eta_tilde(N)
    ch = np.matmul(Dhat,eta)
    return ch

# make a set of coherent states from QFT eigenfunction |tilde eta> and displacement op
# returns an N x N x N 3D complex array
#   the matrix [k,l,:] is the coherent state |k,l>
#  a more efficient version
def coh_Dkl_b(N):
    c_matrix = np.zeros((N,N,N),dtype=complex)
    eta=eta_tilde(N)  #only do this once
    omega = np.exp(2*np.pi*1j/N)  #only do this once
    #omfac = np.power(-omega*k*l/2.0)
    X = Xhat_op(N) # only do this once
    Z = Zhat_op(N) # only do this once
    Zk = np.linalg.matrix_power(Z, 0) # equivalent to identity
    for k in range(N):
        Xl = np.linalg.matrix_power(X, 0) # equivalent to identity
        for l in range(N):
            #Xl = np.linalg.matrix_power(X, l)
            Dhat = np.power(omega,-k*l/2.0) * np.matmul(Zk,Xl)
            #Dhat = Displacement_op(N,k,l)
            c_matrix[k,l,:] = np.matmul(Dhat,eta)
            # if displacement is in p direction then p increases with l
            # and k increases in x direction
            # note that imshow displays arrays with [j,k] with j in y direction
            Xl = np.matmul(X,Xl)
        Zk = np.matmul(Z,Zk)
    return c_matrix
   
# compute Hussimi distribution of an N dimensional state vector psi
# in advance you have computed a set of coherent states with above
#    routine coh_Dkl_b(N)
# the set of coherent states is given via the argument c_matrix
#    c_matrix should be an NxNxN matrix
# the routine returns an NxN real matrix which is the Hussimi function
#   note that index order is p,x  so that y would be first index, as expected for plotting with imshow
def Husimi(psi,c_matrix):
    N = len(psi)
    H_matrix = np.zeros((N,N))
    cshape = c_matrix.shape
    if (cshape[0] != N):
        print('c_matrix is wrong dimension')
    for k in range(N):
        for l in range(N):
            co_kl = np.squeeze(c_matrix[k,l,:])  #coherent state |k,l> -p,x
            mag = np.vdot(psi,co_kl) # dot product
            mag = np.absolute(mag) # is real
            H_matrix[k,l] = mag**2

    return H_matrix


# routines for showing eigenvalues of a unitary operator 
# also the QFT

# arguments: w an array of complex eigenvalues, assumed roots of unity
# returns: an array of normalized differences between the phases of the eigenvalues 
twopi = np.pi*2
def compute_s(w):
    phi_arr = np.angle(w) # get the phase angles in [-pi,pi] of complex numbers
    # this ignores magnitude of w if it is not on the unit circle 
    phi_arr_sort = np.sort(phi_arr)  # put in order of increasing phase
    phi_shift = np.roll(np.copy(phi_arr_sort),1)
    dphi = (phi_arr_sort - phi_shift + np.pi)%twopi - np.pi  # takes care of 2pi shift
    mu_dphi = np.mean(dphi)  # find mean value of dphi 
    sarr = dphi/mu_dphi  # this is s /<s>
    return sarr  # returns normalized phase differences 

# sort eigenvalues and eigenvectors in order of increasing phase
# eigenvectors from np.linalg.eig are in form vr[:,j]
def esort_phase(w,vr):
    phi_arr = np.angle(w) # get the phase angles in [-pi,pi] of complex numbers
    iphi = np.argsort(phi_arr) # in order of increasing phase
    wsort = w[iphi]
    vrsort = np.copy(vr)*0.0
    for i in range(len(w)):
        vrsort[:,i] = vr[:,iphi[i]]
    #vrsort = vr[:,iphi]  # does this work? I think so! yes
    return wsort,vrsort

# create a probability vector for the i-th eigenvector 
# vr is an array of eigenvectors 
def get_probs(vr,i):
    v = vr[:,i]
    probi = np.real(v*np.conjugate(v)) # probability vector 
    return probi  # should be n dimensional real vector 


# fill matrices with Discrete Fourier transform, returns 2 nxn matrices
# both Q_FT and Q_FT^dagger which is the inverse 
def QFT(n):
    omega = np.exp(2*np.pi*1j/n)
    Q        = np.zeros((n,n),dtype=complex)  # QFT
    Q_dagger = np.zeros((n,n),dtype=complex) 
    for j in range(n):
        for k in range(n):
            Q[j,k] = np.power(omega,j*k)  # not 100% sure about sign here! 
            Q_dagger[j,k] = np.power(omega,-j*k)
    Q        /= np.sqrt(n)  #normalize
    Q_dagger /= np.sqrt(n)
    return Q,Q_dagger
    
    
# compute the expectation value of operator, using a state vector evec
def exp_val(evec,op):
    zvec = np.matmul(op,evec)   # compute op|evec>
    w = np.vdot(evec,zvec)   # complex dot product, evec is conjugated prior to doing the dot product
    return w

# sort a set of eigenvalues and eigenfunctions according to the expectation value of an operator
# returns expectation values and dispersions
# arguments:
#  w is list of eigenvals
#  vr is list of eigenvectors
#  op is operator that you want to use to sort
# returns:
#   wsort: list of eigenvalues but these are in order of exp of op
#     computed with the eigenvecs
#   vrsort: list of eigenvecs but in order of <op>
#   expsort:  the expectation values <op>
#   sigsort:  the dispersion values (not standard deviations)
#
def esort_op(w,vr,op):
    n = len(w)
    exp_arr = np.zeros(n,dtype=complex) # to store the expectation values  <op>
    sig_arr = np.zeros(n,dtype=complex) # to store <op^2> - <op>^2
    for i in range(n):
        mu =  exp_val(np.squeeze(vr[:,i]), op)  #h
        mu2 = exp_val(np.squeeze(vr[:,i]), np.matmul(op,op)) #h^2
        sig2 = mu2-mu*mu # compute dispersion too
        exp_arr[i] = mu
        sig_arr[i] = sig2  # notice is dispersion not std
        # compute the expectation values of the operator for every eigenfunction
    iphi = np.argsort(np.real(exp_arr)) # sort in order of increasing expectation value
    wsort = w[iphi]  # sort the eigenvalues
    expsort = exp_arr[iphi] # sort the expectation values
    sigsort = sig_arr[iphi] # sort the dispersions
    vrsort = np.copy(vr)*0.0 # to store the eigenfunctions in order of expectation vals
    for i in range(len(w)):
        vrsort[:,i] = vr[:,iphi[i]]  # sort the eigenfunctions which are vr[:,j]

    return wsort,vrsort,expsort,sigsort

# $\hat h_0 = a(1 - cos \hat p) - \epsilon \cos \hat \phi$
# computes the operator hat h_0, unperturbed hamiltonian
# much better routine
def hat_h_0_new(N,a,eps):
    return a*(np.identity(N) - cos_p_op(N)) - eps*cos_phi_op(N)
    
# $\hat h_0 = a(1 - cos \hat (p-b)) - \epsilon \cos \hat \phi$
#           = a(1 - cos \hat p \cos b - \sin \hat p \sin b) - \epsilon \cos \hat \phi
# computes the operator hat h_0, unperturbed hamiltonian with b term in it
def hat_h_0_with_b(N,a,b,eps):
    return a*(np.identity(N) - cos_p_op(N)*np.cos(b) - sin_p_op(N)*np.sin(b)) \
        - eps*cos_phi_op(N)
    
    
# compute the propagator U across tau =0 to 2 pi
# trying a somewhat more efficient way to do this!
#  arguments:
#     N      : size of discrete quantum space
#     ntau   : how many Trotterized steps to take
#     a,b,eps,mu,mup,taushift      : parameters of classical model, all unitless
#  returns:
#    Ufinal  : The propagator hat U_T  (NxN complex matrix)
#    w       : vector of eigenvalues of U in order of increasing phase
#    vr      : vector of associated eigenfunctions of U
#        vr[:,j] is the eigenvector with eigenvalue w[j]
# note we have shifted indexing so that phi=0, p=0 is in the center of the 2d arrays
shift_by_pi = 0  # for shifting by pi
def U_prop2(N,ntau,a,b,eps,mu,mup,taushift):
    DLambda_A =np.zeros((N,N),dtype=complex)  # storing diagonal matrix for momentum part
    DLambda_Ah =np.zeros((N,N),dtype=complex)  # storing diagonal matrix for momentum part
    DLambda_Ahm =np.zeros((N,N),dtype=complex)  # storing diagonal matrix for momentum part
    DLambda_B =np.zeros((N,N),dtype=complex)  # storing diagonal matrix for phi part
    U_final  =np.zeros((N,N),dtype=complex)  # final propagator
    #Efac = N**2/(4.0*np.pi*ntau)    # includes dtau , is wrong
    Efac = N/ntau    # Bohr Sommerfeld type quantization gives this
    # this is L_0/hbar x 2pi/ntau = N/2pi x 2pi/ntau = N/ntau, is correct , includes dtau
    
    dtau = 2*np.pi/ntau  # step size
    Q_FT,Q_FT_dagger = QFT(N)  # need only be computed once
    # why compute in fourier space if we could use clock and shift for 1 - cos p?
    # b=0 it is simple and if b ne 0 then we can expand
    # as cos p and sin p and both can be written in terms of X, Xhat
    # The real reason we don't bother is that these ops only need be created
    # once here so this part of the routine need not be efficient
    # note you cannot take an exponential of a matrix with np.exp, unless it is diagonal
    for k in range(N):  #compute ahead of time, fill diagonals
        pk = 2*np.pi*k/N - shift_by_pi  # if you want to shift so 0,0 in center of arrays
        DLambda_Ah[k,k] = np.exp(-0.5j*Efac*a*(1.0 - np.cos(pk-b))) #diagonal in Fourier basis
        DLambda_A[k,k]  = np.exp(  -1j*Efac*a*(1.0 - np.cos(pk-b)))
        DLambda_Ahm[k,k]= np.exp( 0.5j*Efac*a*(1.0 - np.cos(pk-b)))
        # these are half step and full step for the momentum part
        
    LAh     = np.matmul(Q_FT,np.matmul(DLambda_Ah ,Q_FT_dagger))  # transfer basis
    LA      = np.matmul(Q_FT,np.matmul(DLambda_A  ,Q_FT_dagger))
    LAh_inv = np.matmul(Q_FT,np.matmul(DLambda_Ahm,Q_FT_dagger))
    
    U_final = LAh;  # half step at the beginning
    
    phivec = np.arange(N)*2*np.pi/N - shift_by_pi  # an array of phis
        
    for i in range(ntau): # each dtau
        tau = i*dtau + taushift # time of propagator shifted by taushift
          
        # replacing a forloop with ability to make a matrix putting  1d array on diagonal
        diagonal = np.exp(1j*Efac*(\
                    eps*np.cos(phivec)+mu*np.cos(phivec-tau)+mup*np.cos(phivec+tau) ))
        # create diagonal, is a 1d array
        # note sign of 1j (-1 * -1 = 1)
        DLambda_B = np.diag(diagonal) # create matrix from a 1d array
        
        U_final = np.matmul(LA,np.matmul(DLambda_B,U_final))
        
    U_final = np.matmul(LAh_inv,U_final)
    
    (w,vr)=np.linalg.eig(U_final)  # get eigenvalues and eigenvectors
    w_s,vr_s = esort_phase(w,vr)   # sort in order of increasing eigenvalue phase
    return w_s,vr_s,U_final
            
    
# compute and store Husimi distributions for all eigenfunctions in an nxnxn array
# arguments:
#    vr eigenvectors  (nxn) matrix but eigenvectors are [:,j]
#    c_matrix nxn matrix of coherent states, precomputed with Dkl_b(n)
# returns matrix of Husimi distributions
def stor_Hus(vr,c_matrix):
    vshape = vr.shape; cshape = c_matrix.shape
    n =vshape[0]
    if (cshape[0]  != n):
        print('c_matrix wrong size')
        return 0
    Hmatrix_big = np.zeros((n,n,n))  # allocate memory
    for k in range(n):   # loop over eigenstates
        Hmatrix_big[:,:,k]= Husimi(np.squeeze(vr[:,k]),c_matrix)
    return Hmatrix_big

# show Husimi functions for all eigenfunctions
# arguments:
#  Hmatrix_big: matrix of husimi functions, precomputed
#  pcolorbar = True/False, whether or not you want to show the colorbar
import matplotlib.ticker as mticker
def show_Hus(Hmatrix_big,froot,pcolorbar):
    if (hasattr(Hmatrix_big, "__len__") == False):
        exit
    hshape = Hmatrix_big.shape;
    n =hshape[0]
    sn = int(np.sqrt(n))
    if (n - sn*sn>0):
        sn+=1;   #increase number of square so that all states are displayed

    fig,axarr = plt.subplots(sn,sn,figsize=(5.1,5.0),sharex=True,sharey=True,dpi=200)
    plt.subplots_adjust(hspace=0,wspace=0,left=0.08,right=1.0,top=0.95,bottom=0.0);
    axarr[0,0].set_xticks([])
    axarr[0,0].set_yticks([])
    zmax = 1/sn  # could be adjusted!!!! or an argument
    
    ax=fig.add_axes([0,0,1,1],frame_on = False)
    ax.set_xticks([])
    ax.set_yticks([])
    if (n < 200):
        ax.text(0.00,0.94,'b)',transform=ax.transAxes,fontsize=26)
    
    for i in range(sn):
        for j in range(sn):
            axarr[i,j].set_aspect('equal')
    nhalf = (int)(n/2)
    
    for i in range(sn):   # plots are in y direction from top to down
        for j in range(sn):  # plots in x direction from left to right
            k = i*sn + j  # incrementing horizontally first
            if (k < n):
                Hmatrix_k = np.squeeze(Hmatrix_big[:,:,k])
                Hmatrix_b = np.roll(Hmatrix_k, nhalf, axis=0) # shift 0,0 to center of image
                Hmatrix = np.roll(Hmatrix_b, nhalf, axis=1)
                im=axarr[i,j].imshow(Hmatrix,origin='lower',vmin=0,\
                                     vmax=zmax,cmap='terrain') # flips y direction of display
                #im=axarr[i,j].imshow(Hmatrix,vmin=0,vmax=zmax,cmap='terrain')
                if (k==0):
                    im0 = im
    
    if (pcolorbar==True):
        cax=fig.add_axes([0.055,0.03,0.02,0.5])
        cbar = plt.colorbar(im0,cax=cax,ticks = [0,zmax],location='left',format=mticker.FixedFormatter(['0', '']))
        ccc = '{:.3e}'.format(zmax)
        cax.text(0.5,zmax*1.02,ccc,fontsize=8,rotation='vertical',va='bottom',ha='center')

    #axarr[0,0].plot([0],[0],'ro')
    
    if (len(froot)>2):
        ofile = froot + '_Hus.png'
        plt.savefig(ofile,dpi=200)
    plt.show()


#sim2v.Hus(True) # show husimi function of eigenvecs
#sim4.Hus(True) # show husimi function of eigenvecs






