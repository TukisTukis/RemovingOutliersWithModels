%====================================================================
% Linear inverse problem parameter estimate using different weights.
%
% Implementation that generates Figure 2 in the article:
% "Outlier removal for improved source estimation
%  in atmospheric inverse problems",
%  
%  By M. Martinez-Camara, A. Stohl and M. Vetterli, ICASSP 2014
%
%  Marta Martinez-Camara, LCAV, EPFL, 2013
%====================================================================
% Requirements: cvx

clear; clc;

% suppress output from cvx
cvx_quiet(true); 

% set the random seed to the clock (independent results each time this script runs)
rng('shuffle'); 

% load ETEX data
load ('./Data/etex_dataset.mat')

% Model using mateo dataset 1
M = 1e13*M_era40; 

[m,n] = size(M);

% scaling measurements to avoid numerical errors
y = 1e13*measurements;

source = x;
syntMeas = M*source;

% index of the measurements that we use
ind = ones(m,1); 

% number of rounds
nRounds = 10; 

% percentage of measurements we keep in each iteration
outKeep = 0.98; 

% measurements we use in each round
newy = y; 
newM = M; 

% error if we use all the measurements
initial_residuals = y - syntMeas;

% here we'll keep the original errors
new_residuals = initial_residuals; 
% how many measurements we keep
nkeep = m; 
% best mse for each round
MSE = zeros(nRounds,1); 
INX = zeros(nRounds,1); 

for idx_rnd = 1:nRounds
    % in every round...
    
    nlbds = 20;
    lambda = logspace(-1,2,nlbds);
    xNNLS = zeros (n,nlbds);
    
    % search the optimum reg. parameter for every round
    for idx_lb = 1:nlbds
        xNNLS(:,idx_lb) =  solveInverse(newM,newy,lambda(idx_lb),'NNCTIK');
    end
    mse =(mean((xNNLS-repmat(source,1,nlbds)).^2,1));
    % min MSE for this round
    [mE,mInx] = min(mse);
    INX(idx_rnd) = mInx; 
    MSE(idx_rnd) = mE; 
   
  
    errorR = newy - newM*xNNLS(:,mInx);
    
    [eOrdered,iOrdered] = sort(abs(new_residuals));
    
    % we remove 20 measurements each round
    nkeep = nkeep - 20;
    
    % indexes of the meas. with the smallest residuals
    ind = iOrdered (1:nkeep); 
    newy = newy(ind);
    newM = newM(ind,:);
    new_residuals = new_residuals(ind);
    

end

% plot results

% nice big figure
figure('position',[100,100,900,500]); 

plot(0:20:180,MSE*100/MSE(1),'ko-')
% use large fonts
set(gca,'fontsize',14); 
% easier to read off values
grid on; 
% label the x axis
xlabel('Number of measurements removed'); 
% label the y axis
ylabel('MSE of reconstruction (%)')

