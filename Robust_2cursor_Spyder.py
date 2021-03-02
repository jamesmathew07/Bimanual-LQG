# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 11:08:14 2021
translated code from matlab author :anthony savocco
@author: jmathew
"""
#------------------------------------------------------------------------#
# based on the papers:
#
# 1) "A very fast time scale of human motor adaptation: Within movement 
# adjustments of internal representations during reaching", Crevecoeur
# 2020.
#
# 2) "Optimal Task-Dependent Changes of Bimanual Feedback Control and 
# Adaptation", Diedrichsen 2007.


#clear workspace
def clear_all():
    """Clears all the variables from the workspace of the spyder application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]
if __name__ == "__main__":
    clear_all()
    
    
# import libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import pandas as pd
import copy
import warnings
warnings.filterwarnings("ignore")

# #---PARAMETERS---# ## 

m              = 2.5;  # [kg]
k              = 0.1;  # [Nsm^-1]
tau            = 0.1;  # [s]
delta          = 0.01; # [s]  
theta          = 15;   # [N/(m/s)] - coeff pert force (F = +-L*dy/dt)
alpha          = np.array([[1000, 1000, 20, 20, 0, 0]]);# [PosX, PosY, VelX, VelY, Fx, Fy]
learning_rates = np.array([[0.1,0.1]]);# [right left]
coeffQ         = 1;      # increase or decrease Q matrix during trials
time           = 0.6;    # [s] - experiment 1 - reaching the target
stab           = 0.01;   # [s] - experiment 1 - stabilization 
nStep          = round((time+stab)/delta)-1;
N              = round(time/delta);

# Protocol parameters

right_perturbation = 'BASELINE';     # CCW, CW or BASELINE (no FFs)
left_perturbation  = 'CCW';          # CCW, CW or BASELINE (no FFs)
numoftrials        = 20;        # number of trials 
catch_trials       = 0;         # number of catch trials

## #---SYSTEM CREATION---# ##

# STATE VECTOR REPRESENTATION:
# (1-6)  RIGHT HAND x, y, dx, dy, fx and fy
# (7-12) LEFT HAND  x, y, dx, dy, fx and fy

xinit  = np.array([.06, 0, 0, 0, 0, 0, -.06, 0, 0, 0, 0, 0]);    # [right left]
xfinal = np.array([.06, .15, 0, 0, 0, 0, -.06, .15, 0, 0, 0, 0]);# [right left] 

#---System---#
A = np.array([[0, 0, 1, 0, 0, 0], \
             [ 0, 0, 0, 1, 0, 0], \
             [0, 0, -k/m, 0, 1/m, 0], \
             [0, 0, 0, -k/m, 0, 1/m], \
             [0, 0, 0, 0, -1/tau, 0], \
	         [0, 0, 0, 0, 0, -1/tau]]);
A = scipy.linalg.block_diag(A,A);	 
B = np.array([[0,0],\
              [0,0],\
              [0,0],\
              [0,0],\
              [1/tau,0],\
              [0,1/tau]])    
B   = scipy.linalg.block_diag(B,B);

ns  = np.size(A,0)
nc  = np.size(B,1)

A_hat = copy.deepcopy(A);
DA    = (A-A_hat)*delta; # Used when there is a model error
A     = np.eye(np.size(A,0))+delta*A;
A_hat = np.eye(np.size(A_hat,0))+delta*A_hat;
B     = delta*B;

# Observability Matrix
H = np.eye(np.size(A,0));
E = np.eye(ns,1).T;          #See Basar and Tamer, pp. 171

## #---COST FUNCTION---# ##

Q  = np.zeros(((np.size(A,0),np.size(A,1),nStep)));
# M  = copy.deepcopy(Q);
# TM = copy.deepcopy(Q);
Id = np.eye((ns));

#weights of parameter
runningalpha = np.zeros((ns,nStep)); 
for i in np.arange(nStep):
    fact = np.power(np.amin((1,((i+1)*delta/time))),6);
    temp = np.array([np.power(fact*10,8),np.power(fact*10,8),np.power(fact*10,4),np.power(fact*10,4), 1, 1]);
    runningalpha[:,i] = np.hstack([temp, temp]);
    
#Filling in the cost matrices
for j in np.arange(nStep):
    for i in np.arange(ns):
        Q[:,:,j] = Q[:,:,j] + runningalpha[i,j]* (Id[:,[i]] @ Id[:,[i]].T);       

#Signal Dependent Noise
nc   = np.size(B,1);
Csdn = np.zeros(((np.size(B,0),nc,nc)));
for i in np.arange(nc):
    Csdn[:,i,i] = 0.1*B[:,i];    


M  = copy.deepcopy(Q);
TM = copy.deepcopy(Q);
D  = np.eye((ns));

# Implementing the backwards recursions

M[:,:,-1] = Q[:,:,-1];
L          = np.zeros(((np.size(B,1),np.size(A,0),nStep-1)));  # Optimal Minimax Gains
Lamda     = np.zeros(((np.size(A,0),np.size(A,1),nStep-1)));

# Optimization of gamma
gamma      = 50000;
minlambda  = np.zeros((nStep-1,1));
gammaK     = 0.5;
reduceStep = 1;
positive   = 0; #false
relGamma   = 1;

while (relGamma > .001 or not positive):

    for k in np.arange(nStep-2,0,-1):
        # Minimax Feedback Control
        TM[:,:,k]    = gamma**2 *np.eye(np.size(A,0))- D.T @ M[:,:,k+1] @ D;
        minlambda[k] = np.amin(np.linalg.eigvals(TM[:,:,k]));

        Lamda[:,:,k] = np.eye(np.size(A_hat,0)) + (B@B.T- (1/(gamma**2))*(D@D.T)) @ M[:,:,k+1];
        M[:,:,k]      = Q[:,:,k] + A_hat.T @ np.linalg.inv((np.linalg.inv(M[:,:,k+1]) + B@B.T-(1/(gamma**2)) * D@D.T)) @ A_hat;
        L[:,:,k]      = B.T @ M[:,:,k+1] @ np.linalg.inv(Lamda[:,:,k]) @ A_hat;

    oldGamma = copy.deepcopy(gamma);

    if np.amin(np.real(minlambda)) >= 0:
        gamma    = (1-gammaK)*gamma;
        relGamma = (oldGamma-gamma)/oldGamma;
        positive = 1;

    elif np.amin(np.real(minlambda)) < 0:
        gamma      = (1/(1-gammaK))*gamma;
        reduceStep = reduceStep + 0.5;
        relGamma   = -(oldGamma-gamma)/oldGamma;
        gammaK     = gammaK**reduceStep;
        positive   = 0;

# #---SIMULATION---# ##

# Add perturbation (curl FFs) to the matrix A
if right_perturbation == 'CCW':
        A[2,3] = -delta*(theta/m);
        A[3,2] = delta*(theta/m);
elif right_perturbation == 'CW':
        A[2,3] = delta*(theta/m);
        A[3,2] = -delta*(theta/m);
elif right_perturbation == 'BASELINE':
        A[2,3] = 0;
        A[3,2] = 0;
else:
        print('The perturbation choice is incorrect !')


if left_perturbation == 'CCW':
        A[8,9] = -delta*(theta/m);
        A[9,8] = delta*(theta/m);
elif left_perturbation ==  'CW':
        A[8,9] = delta*(theta/m);
        A[9,8] = -delta*(theta/m);
elif left_perturbation ==  'BASELINE':
        A[8,9] = 0;
        A[9,8] = 0;
else:
        print('The perturbation choice is incorrect !')


# Initialization simulation

 # Robust
x    = np.zeros(((ns,nStep+1,numoftrials)));
xhat = copy.deepcopy(x);

control     = np.zeros(((nc,nStep,numoftrials)));    # Initialize control
avControl   = np.zeros(((nc,nStep)));                # Average Control variable

# Random indexes for catch trials
# catch_trials_idx = [];

# if catch_trials ~= 0
#     while length(catch_trials_idx) ~= catch_trials
#         random = randi(numoftrials, 1, 1); 
#         catch_trials_idx = [catch_trials_idx random];
#         catch_trials_idx = unique(catch_trials_idx);
#     end
# end

catch_trials_idx = 10;

for p in np.arange(numoftrials):

    # Robust
    x[:,0,p]     = xinit - xfinal;
    xhat[:,0,p]  = x[:,0,p];
    u            = np.zeros((nStep-1,np.size(B,1))); # size(B,2) is the control dimension
    w            = np.zeros((ns,1));
    Oxi          = 0.001*B@B.T;
    Omega        = np.eye(6)*Oxi[4,4];

    #Parameters for State Estimation
    Sigma        = np.zeros(((ns,ns,nStep)));
    Sigma[:,:,0] = np.eye(ns)*0.001;
    SigmaK       = copy.deepcopy(Sigma);

    for i in np.arange(nStep-1):

        #--- Robust ---#
        A_old = copy.deepcopy(A);
        
        if k == catch_trials_idx:
            A[2,3]  = 0;
            A[8,9]  = 0;
            A[3,2]  = 0;
            A[9,8]  = 0;
        

        sensoryNoise = (np.random.multivariate_normal(np.zeros(np.size(Omega,0)),Omega)).T;
        sensoryNoise = np.hstack([sensoryNoise, sensoryNoise]);
        motorNoise   = (np.random.multivariate_normal(np.zeros(np.size(Oxi,0)),Oxi)).T;

        #MINMAX HINFTY CONTROL ------------------------------------------------
        #Riccati Equation for the State Estimator
        Sigma[:,:,i+1] = A_hat @ np.linalg.inv(np.linalg.inv(Sigma[:,:,i]) + (H.T * np.linalg.inv(E@E.T)) @ H - (1/gamma**2) * Q[:,:,i]) @ A_hat.T + D @ D.T;

        #Feedback Equation
        yx = H @ x[:,i,p] + sensoryNoise;

        #Minmax Simulation with State Estimator
        u[i,:] = -B.T @ np.linalg.inv(np.linalg.inv(M[:,:,i+1]) + B@B.T - (1/gamma**2) * (D@D.T)) @ A_hat @ \
                  np.linalg.inv(np.eye(ns)-(1/gamma**2)*Sigma[:,:,i] @ M[:,:,1]) @ xhat[:,i,p];

        #Signal Dependent Noise - Robust Control
        sdn    = 0;

        for isdn in np.arange(nc):
            sdn = sdn + np.random.normal(0,1)*Csdn[:,:,isdn] @ (u[i,:].T);


        xhat[:,i+1,p] = A_hat @ xhat[:,i,p] + B  @ u[i,:].T   \
                        + A_hat @ np.linalg.inv(np.linalg.inv(Sigma[:,:,i]) + (H.T * np.linalg.inv(E@E.T)) @ H - (1/gamma**2) * Q[:,:,i]) \
                        @ ((1/gamma**2)* Q[:,:,i] @ xhat[:,i,p] \
                        + (H.T * np.linalg.inv(E@E.T))  @ (yx - H @ xhat[:,i,p]));

        # Minmax Simulation
        DA         = A - A_hat;          
        wx         = DA @ x[:,i,p]; # Non zero if there is a model error. 
        x[:,i+1,p] = A_hat @ x[:,i,p] + B @ u[i,:].T + D @ wx + motorNoise + sdn;

        # Update the A matrix
        
        # # Update the A matrix (see eq.(9))
        eps          = x[0:5,i+1,p]- xhat[0:5,i+1,p];
        
        theta_up_R    = A_hat[2,3];
        dzhat_dL      = np.zeros(5);
        dzhat_dL[2]   = xhat[3,i+1,p];# x_hat check
        theta_up_R    = theta_up_R + learning_rates[0,0]*dzhat_dL @ eps;
        A_hat[2,3]    = theta_up_R;
        
        theta_up_R    = A_hat[3,2];
        dzhat_dL      = np.zeros(5);
        dzhat_dL[3]   = xhat[2,i+1,p];# x_hat check
        theta_up_R    = theta_up_R + learning_rates[0,0]*dzhat_dL @ eps;
        A_hat[3,2]    = theta_up_R;
        
        eps           = x[6:11,i+1,p]- xhat[6:11,i+1,p];
        
        theta_up_L    = A_hat[8,9];
        dzhat_dL      = np.zeros(5);
        dzhat_dL[2]   = xhat[9,i+1,p];
        theta_up_L    = theta_up_L + learning_rates[0,1]*dzhat_dL @ eps;
        A_hat[8,9]    = theta_up_L;
        
        theta_up_L    = A_hat[9,8];
        dzhat_dL      = np.zeros(5);
        dzhat_dL[3]   = xhat[8,i+1,p];
        theta_up_L    = theta_up_L + learning_rates[0,1]*dzhat_dL @ eps;
        A_hat[9,8]    = theta_up_L;

        u_temp         = np.array([u[i,0], u[i,1], u[i,2], u[i,3]]);
        control[:,i,p] = copy.deepcopy(u_temp);
        A              = copy.deepcopy(A_old);
  
    
    avControl = avControl + control[:,:,p]/numoftrials;

## #---GRAPHS---# ##
x2 = copy.deepcopy(x);
for i in np.arange(numoftrials):
    for j in np.arange(N):
        x2[:,j,i]   = x[:,j,i] + np.array([.06, .15, 0, 0, 0, 0, -.06, .15, 0, 0, 0, 0]);
        

for trial in np.arange(numoftrials):
     
    # Position
    plt.subplot(2,2,1)
    
    midx = 0.5*(x2[0,:N,trial]+x2[6,:N,trial]);
    midy = 0.5*(x2[1,:N,trial]+x2[7,:N,trial]);
    
    plt.plot(x2[0,:N,trial], x2[1,:N,trial]);

    plt.plot(x2[6,:N,trial], x2[7,:N,trial]);
    
    plt.plot(midx, midy);
    
    plt.plot(0,0,'ro', mfc='none',markersize=16);
    plt.plot(0,.15,'ro',mfc='none',markersize=16);
    plt.plot(0.06,0,'ro', mfc='none',markersize=16);
    plt.plot(0.06,.15,'ro', mfc='none',markersize=16);
    plt.plot(-0.06,0,'ro', mfc='none',markersize=16);
    plt.plot(-0.06,.15,'ro', mfc='none',markersize=16);
    plt.xlabel('x-coord [m]'); plt.ylabel('y-coord [m]'); plt.title('Robust model - two cursor - trajectories',FontSize= 14);
    # axis([-(max(x(1,:,1)) + 0.04) (max(x(1,:,1)) + 0.04)  -0.01 0.16])

    # # Control
    plt.subplot(2,2,4)
    plt.plot(np.arange(0.01,(nStep+1)*0.01,0.01),control[0,:,trial])
    plt.plot(np.arange(0.01,(nStep+1)*0.01,0.01),avControl[0,:],'k',Linewidth=2)
    plt.xlabel('Time [s]'); plt.ylabel('Control [Nm]'); plt.title('Control Vector - Right',FontSize=14);
    #axis square

    plt.subplot(2,2,3)
    plt.plot(np.arange(0.01,(nStep+1)*0.01,0.01),control[2,:,trial]);
    plt.plot(np.arange(0.01,(nStep+1)*0.01,0.01),avControl[2,:],'k',Linewidth=2)
    plt.xlabel('Time [s]'); plt.ylabel('Control [Nm]'); plt.title('Control Vector - Left',FontSize=14);
    # #axis square

    # #input(' ');    


