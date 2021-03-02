
#translated code from matlab 
#@author :anthony savocco
#@author: jmathew

#------------------------------------------------------------------------#
# based on the papers:
#
# 1) "A very fast time scale of human motor adaptation: Within movement 
# adjustments of internal representations during reaching", Crevecoeur
# 2020.
#
# 2) "Optimal Task-Dependent Changes of Bimanual Feedback Control and 
# Adaptation", Diedrichsen 2007.


# import libraries
library(MASS)
library(magic)
library(matlab)
rm(list = ls());
#dev.off();
#source("C:/Users/jmathew/Dropbox (INMACOSY)/James-UCL/LQG/OFG.R")


# #---PARAMETERS---# ## 

m              = 2.5;  # [kg]
k              = 0.1;  # [Nsm^-1]
tau            = 0.1;  # [s]
delta          = 0.01; # [s]  
theta          = 15;   # [N/(m/s)] - coeff pert force (F = +-L*dy/dt)
alpha          = matrix(c(1000, 1000, 20, 20, 0, 0),nrow=1);# [PosX, PosY, VelX, VelY, Fx, Fy]
learning_rates = matrix(c(.1 ,.1),nrow=1);# [right left]
coeffQ         = 1;      # increase or decrease Q matrix during trials
time           = 0.6;    # [s] - experiment 1 - reaching the target
stab           = 0.01;   # [s] - experiment 1 - stabilization 
nStep          = round((time+stab)/delta)-1;
N              = round(time/delta);

# Protocol parameters

right_perturbation = 'BASELINE';     # CCW, CW or BASELINE (no FFs)
left_perturbation  = 'CCW';          # CCW, CW or BASELINE (no FFs)
numoftrials        = 15;        # number of trials 
catch_trials       = 0;         # number of catch trials

## #---SYSTEM CREATION---# ##

# STATE VECTOR REPRESENTATION:
# (1-6)  RIGHT HAND x, y, dx, dy, fx and fy
# (7-12) LEFT HAND  x, y, dx, dy, fx and fy

xinit  = matrix(c(.06, 0, 0, 0, 0, 0, -.06, 0, 0, 0, 0, 0),ncol=1);    # [right left]
xfinal = matrix(c(.06, .15, 0, 0, 0, 0, -.06, .15, 0, 0, 0, 0),ncol=1);# [right left] 

#---System---#
A = t(matrix(c(0, 0, 1, 0, 0, 0,
               + 0, 0, 0, 1, 0, 0, 
               + 0, 0, -k/m, 0, 1/m, 0, 
               + 0, 0, 0, -k/m, 0, 1/m, 
               + 0, 0, 0, 0, -1/tau, 0, 
               + 0, 0, 0, 0, 0, -1/tau), ncol = 6))

A = adiag(A,A);	 
B = t(matrix(c(0,0,
               + 0,0,
               + 0,0,
               + 0,0,
               + 1/tau,0,
               + 0,1/tau), nrow =2)); 
B = adiag(B,B);

ns  = nrow(A)
nc  = ncol(B)

A_hat = A;
DA    = (A-A_hat)*delta; # Used when there is a model error
A     = diag(ns)+delta*A;
A_hat = diag(nrow(A_hat))+delta*A_hat;
B     = delta*B;

# Observability Matrix
H = diag(nrow(A));
E = t(diag(ns)[1,]);          #See Basar and Tamer, pp. 171

## #---COST FUNCTION---# ##

Q  =  array(0,dim =c(nrow(A),ncol(A),nStep))  
Id = diag(ns);

#weights of parameter
runningalpha = array(0,dim =c(ns,nStep)); 
for (i in c(1:nStep))
{
  fact = min(1,(i*delta/time))^6;
  temp = c(fact*10^8, fact*10^8, fact*10^4, fact*10^4, 1, 1)
  runningalpha[,i] = t(c(temp, temp));
}

#Filling in the cost matrices
for (j in c(1:nStep))
{
  for (i in c(1:ns))
  {
    Q[,,j] = Q[,,j] + runningalpha[i,j]* (Id[,i] %*% t(Id[,i]));     
  }
}

#Signal Dependent Noise
nc   = ncol(B);
Csdn = array(0,dim =c(nrow(B),nc,nc));
for (i in c(1:nc))
{
  Csdn[,i,i] = 0.1*B[,i];    
}

M  = Q;
TM = Q;
D  = diag(ns);

# Implementing the backwards recursions

M[,,dim(M)[3]] = Q[,,dim(Q)[3]];
L              = array(0,dim =c(ncol(B),nrow(A),nStep-1));  # Optimal Minimax Gains
Lambda         = array(0,dim =c(nrow(A),ncol(A),nStep-1));

# Optimization of gamma
gamma      = 50000;
minlambda  = array(0,dim =c(nStep-1,1));
gammaK     = 0.5;
reduceStep = 1;
positive   = 0; #false
relGamma   = 1;

while (relGamma > .001 || !positive)
{
  
  for (k in c((nStep-2):1))
  {
    # Minimax Feedback Control
    TM[,,k]      = gamma^2 *diag(nrow(A))- t(D) %*% M[,,k+1] %*% D;
    ei           = eigen(TM[,,k], only.values = TRUE);
    minlambda[k] = min(ei$values);
    
    Lambda[,,k] = diag(nrow(A_hat)) + (B %*% t(B)- (gamma^-2)*(D%*%t(D))) %*% M[,,k+1];
    M[,,k]      = Q[,,k] + t(A_hat) %*% solve(solve(M[,,k+1]) + B%*%t(B)-(gamma^-2) * D%*%t(D)) %*% A_hat ;
    L[,,k]      = t(B) %*% M[,,k+1] %*% solve(Lambda[,,k]) %*% A_hat;
  }
  
  oldGamma = gamma;
  
  if (min(minlambda) >= 0)
  {
    gamma    = (1-gammaK)*gamma;
    relGamma = (oldGamma-gamma)/oldGamma;
    positive = 1;
  }
  else if (min(minlambda) < 0)
  {
    gamma      = (1/(1-gammaK))*gamma;
    reduceStep = reduceStep + 0.5;
    relGamma   = -(oldGamma-gamma)/oldGamma;
    gammaK     = gammaK^reduceStep;
    positive   = 0;
  }
}

# #---SIMULATION---# ##

# Add perturbation (curl FFs) to the matrix A
if (right_perturbation == 'CCW'){
  A[3,4] = -delta*(theta/m);
  A[4,3] = delta*(theta/m);
} else if (right_perturbation == 'CW'){
  A[3,4] = delta*(theta/m);
  A[4,3] = -delta*(theta/m);
}else if (right_perturbation == 'BASELINE'){
  A[3,4] = 0;
  A[4,3] = 0;
}else {
  print('The perturbation choice is incorrect !')
}

if (left_perturbation == 'CCW')
{
  A[9,10] = -delta*(theta/m);
  A[10,9] = delta*(theta/m);
} else if (left_perturbation ==  'CW'){
  A[9,10] = delta*(theta/m);
  A[10,9] = -delta*(theta/m);
} else if (left_perturbation ==  'BASELINE'){
  A[9,10] = 0;
  A[10,9] = 0;
}else{
  print('The perturbation choice is incorrect !')
}

# Initialization simulationce

# Robust
x    = array(0,dim =c(ns,nStep+1,numoftrials));
xhat = x;

control     = array(0,dim =c(nc,nStep,numoftrials));    # Initialize control
avControl   = array(0,dim =c(nc,nStep));                # Average Control variable

catch_trials_idx = 10;

for (p in c(1:numoftrials))
{
  
  # Robust
  x[,1,p]     = xinit - xfinal;
  xhat[,1,p]  = x[,1,p];
  u            = array(0,dim =c(nStep-1,ncol(B))); # size(B,2) is the control dimension
  w            = array(0,dim =c(ns,1));
  Oxi          = 0.001*B %*% t(B);
  Omega        = diag(6)*Oxi[5,5];
  
  #Parameters for State Estimation
  Sigma        = array(0,dim =c(ns,ns,nStep));
  Sigma[,,1]   = diag(ns)*0.001;
  SigmaK       = Sigma;
  
  for (i in c(1:(nStep-2)))
  {
    
    #--- Robust ---#
    A_old = A;
    
    if (k == catch_trials_idx)
    {
      A[3,4]   = 0;
      A[9,10]  = 0;
      A[4,3]   = 0;
      A[10,9]  = 0;
    }
    
    
    sensoryNoise = t(t(mvrnorm(n = 1, matrix(0,nrow(Omega),1), Omega,  empirical = FALSE)))
    sensoryNoise = rbind(sensoryNoise, sensoryNoise);
    motorNoise   = t(t(mvrnorm(n = 1, matrix(0,nrow(Oxi),1), Oxi,  empirical = FALSE))) 
    
    #MINMAX HINFTY CONTROL ------------------------------------------------
    #Riccati Equation for the State Estimator
    Sigma[,,i+1] = A_hat %*% solve(solve(Sigma[,,i]) + (t(H)  * drop(solve(E%*%t(E)))) %*% H - (gamma^-2) * Q[,,i]) %*% t(A_hat) + D %*% t(D)
    
    #Feedback Eequation
    yx = H %*% x[,i,p] + sensoryNoise
    
    #Minmax Simulation with State Estimator
    u[i,] = - t(B) %*% solve(solve(M[,,i+1]) + B %*% t(B) - (gamma^-2) * (D %*% t(D))) %*% A_hat %*% solve(diag(ns)-(gamma^-2)*Sigma[,,i] %*% M[,,1]) %*% xhat[,i,p]
    
    #Signal Dependent Noise - Robust Control
    sdn    = 0;
    
    for (isdn in c(1:nc))
    {
      sdn = sdn + rnorm(1,0,1)*Csdn[,,isdn] %*% (u[i,]);
    }
    
    xhat[,i+1,p] = A_hat %*% xhat[,i,p] + B  %*% (u[i,])  + A_hat %*%
      solve(solve(Sigma[,,i]) + (t(H) * drop(solve(E%*%t(E)))) %*% H - (gamma^-2) * Q[,,i]) %*% 
      ((gamma^-2)* Q[,,i] %*% xhat[,i,p]  + (t(H) * drop(solve(E %*% t(E))) ) %*% (yx - H %*% xhat[,i,p]));
    
    # Minmax Simulation
    DA         = A - A_hat;          
    wx         = DA  %*% x[,i,p]; # Non zero if there is a model error. 
    x[,i+1,p]  = A_hat  %*% x[,i,p] + B  %*%  u[i,] + D %*% wx + motorNoise + sdn;
    
    # Update the A matrix

    # Update the A matrix (see eq.(9))
    eps          = x[1:6,i+1,p]- xhat[1:6,i+1,p];

    theta_up_R    = A_hat[3,4];
    dzhat_dL      = array(0,dim =c(1,6));
    dzhat_dL[1,3] = xhat[4,i+1,p];# x_hat check
    theta_up_R    = theta_up_R + learning_rates[1]*dzhat_dL %*% eps;
    A_hat[3,4]    = theta_up_R;

    theta_up_R    = A_hat[4,3];
    dzhat_dL      = array(0,dim =c(1,6));
    dzhat_dL[1,4] = xhat[3,i+1,p];# x_hat check
    theta_up_R    = theta_up_R + learning_rates[1]*dzhat_dL %*% eps;
    A_hat[4,3]    = theta_up_R;

    eps           = x[7:12,i+1,p]- xhat[7:12,i+1,p];

    theta_up_L    = A_hat[9,10];
    dzhat_dL      = array(0,dim =c(1,6));
    dzhat_dL[1,3] = xhat[10,i+1,p];
    theta_up_L    = theta_up_L + learning_rates[2]*dzhat_dL %*% eps;
    A_hat[9,10]   = theta_up_L;

    theta_up_L    = A_hat[10,9];
    dzhat_dL      = array(0,dim =c(1,6));
    dzhat_dL[1,4] = xhat[9,i+1,p];
    theta_up_L    = theta_up_L + learning_rates[2]*dzhat_dL %*% eps;
    A_hat[10,9]   = theta_up_L;
    
    u_temp         = c(u[i,1], u[i,2], u[i,3], u[i,4]);
    control[,i,p]  = u_temp;
    A              = A_old;
  }
  
  avControl = avControl + control[,,p]/numoftrials
}

## #---GRAPHS---# ##

translate = repmat(t(t(c(.06, .15, 0, 0, 0, 0, -.06, .15, 0, 0, 0, 0))),1,N,numoftrials);
x2        = x[,1:N,] + translate;

# The following can be used instead of above 2 lnes of code
# x2 = x;
# for (i in c(1:numoftrials))
# {
#   for (j in c(1:N))
#   {
#     x2[,j,i] = x[,j,i] + c(.06, .15, 0, 0, 0, 0, -.06, .15, 0, 0, 0, 0);
#   }
# }



for (trial in c(1:numoftrials))
{
  
  # Position
  par(mfrow=c(2,2))
  
  midx = 0.5*(x2[1,1:N,trial]+x2[7,1:N,trial]);
  midy = 0.5*(x2[2,1:N,trial]+x2[8,1:N,trial]);
  par(mfg=c(1,1));par(new=TRUE);
  plot(x2[1,1:N,trial], x2[2,1:N,trial],type="l",ann=FALSE,xlim=c(-0.10,0.10),ylim=c(-.02,.20));par(new=TRUE);
  plot(x2[7,1:N,trial], x2[8,1:N,trial],type="l",ann=FALSE,xlim=c(-0.10,0.10),ylim=c(-.02,.20));par(new=TRUE);
  plot(midx, midy,type="l",xaxt='n',yaxt='n',ann=FALSE,xlim=c(-0.10,0.10),ylim=c(-.02,.20));par(new=TRUE);
  plot(0,0,col ="red",lwd=8,ann=FALSE,xlim=c(-0.10,0.10),ylim=c(-.02,.20));par(new=TRUE);
  plot(0,.15,col ="red",lwd=8,ann=FALSE,xlim=c(-0.10,0.10),ylim=c(-.02,.20));par(new=TRUE);
  plot(0.06,0,col ="red",lwd=8,ann=FALSE,xlim=c(-0.10,0.10),ylim=c(-.02,.20));par(new=TRUE);
  plot(0.06,.15,col ="red",lwd=8,ann=FALSE,xlim=c(-0.10,0.10),ylim=c(-.02,.20));par(new=TRUE);
  plot(-0.06,0,col ="red",lwd=8,ann=FALSE,xlim=c(-0.10,0.10),ylim=c(-.02,.20));par(new=TRUE);
  plot(-0.06,.15,col ="red",lwd=8,ann=FALSE,xlim=c(-0.10,0.10),ylim=c(-.02,.20));par(new=TRUE);
  
  
  # Control
  par(mfg=c(1,2));par(new=TRUE);
  plot(c(.01:(nStep)*.01:.01),control[1,,trial],type="l",ann=FALSE,ylim=c(-130,130));par(new=TRUE);
  plot(c(.01:(nStep)*.01:.01),avControl[1,],type="l",col="magenta",lwd=2,ann=FALSE,ylim=c(-130,130));par(new=TRUE);
  
  par(mfg=c(2,1));par(new=TRUE);
  plot(c(.01:(nStep)*.01:.01),control[3,,trial],type="l",ann=FALSE,ylim=c(-130,130));par(new=TRUE);
  plot(c(.01:(nStep)*.01:.01),avControl[3,],type="l",col="magenta",lwd=2,ann=FALSE,ylim=c(-130,130));par(new=TRUE);
  
}
par(mfg=c(1,1));par(new=TRUE);
title(main = 'Robust model - two cursor - trajectories',xlab="x-coord [m]", ylab="y-coord [m]");
par(mfg=c(1,2));par(new=TRUE);
title(main ='Control Vector - Right',xlab='Time [s]', ylab='Control [Nm]');
par(mfg=c(2,1));par(new=TRUE);
title(main ='Control Vector - Left',xlab='Time [s]', ylab='Control [Nm]')
