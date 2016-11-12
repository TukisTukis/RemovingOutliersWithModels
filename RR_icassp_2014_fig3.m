%====================================================================
% Linear inverse problem parameter estimate using different weights.
%
% Implementation that generates Figure 3 in the article:
% "Outlier removal for improved source estimation
%  in atmospheric inverse problems",
%  
%  By M. Martinez-Camara, A. Stohl and M. Vetterli, ICASSP 2014
%
%  Marta Martinez-Camara, LCAV, EPFL, 2013
%====================================================================
% Requirements: cvx

% tabula rasa
clear; close all; clc; 

% always generate new random numbers
rng('shuffle'); 

% how many experiments do we want to carry out
realizations = 1000; 

% generate synthetic data

% We use the ratio between rows and columns in the ETEX dataset
r = ceil(3102/120);

% set number of columns
n = 10;

% set number of rows
m = r*n;

% generate random matrix
A = rand(m,n);

% set source
x = zeros(n,1); 
x(3:7) = 1;

% different ratio of outliers in the data to test
outRatio = 0:0.05:0.3;
nOut = numel(outRatio);

% store results
MSE = zeros(realizations,numel(outRatio),3);

for j = 1:numel(outRatio)
  % for every ratio of outliers to test...  
  for i = 1:realizations
      % for each realization...
      
      % sparse uniformly distributed random matrix
      N1 =  sprand(m,n,outRatio(j)); 
      % sparse uniformly distributed random matrix
      N2 =  sprand(m,n,outRatio(j)); 
      % noisy model matrix 1
      M1 = A + N1; 
      % noisy model matrix 2
      M2 = A + N2; 
      % create measurements 
      y = A*x; 
      
      % solution without using weights
      x_noweights = pinv(M1)*y;
      
      % differences between the two model matrices
      Dff = M1 - M2;
      
      % accumulated differences in each row
      Dac = sum (abs(Dff),2); 
      
      % exponential weights
      ws = exp(-Dac);
      W = diag(ws);
      % solution using exp. weights
      xW_exp = pinv(W*M1)*W*y; 

      % thresholding weights
      th = 1.5;
      % which are the indexed of good measurements
      goodInx = Dac < th; 
      % build linear weights
      wsl = (1/(Dac + 1).^2)'; 
      wsl(goodInx) = 1;
      Wl = diag(wsl);
      % solution using th. weights
      xW_th = pinv(Wl*M1)*Wl*y; 
      
      % compute mse of each solution
      mse_noweights = mean((x-x_noweights).^2);
      mse_exp = mean((x-xW_exp).^2);
      mse_th = mean((x-xW_th).^2);
      
      % save it 
      MSE(i,j,1) = mse_noweights;
      MSE(i,j,2) = mse_exp;
      MSE(i,j,3) = mse_th;
  end
end

% take averages over all the realizations
average_MSE_nw = mean(MSE(:,:,1),1);
average_MSE_exp = mean(MSE(:,:,2),1);
average_MSE_th = mean(MSE(:,:,3),1);

% plot results

% nice big figure
figure('position',[100,100,900,500]); 

% plot lines
plot(outRatio,average_MSE_nw, 'b--', 'LineWidth',2)
hold on
plot(outRatio,average_MSE_exp,'r-','LineWidth',2)
plot(outRatio,average_MSE_th,'k-.','LineWidth',2)

% use large fonts
set(gca,'fontsize',14); 
% easier to read off values
grid on; 
% label the x axis
xlabel('Ratio of errors in the matrix'); 
% label the y axis
ylabel('MSE of reconstruction')
% label the curves
legend('without weights','exp. weights','th. weights','location','northwest'); 









