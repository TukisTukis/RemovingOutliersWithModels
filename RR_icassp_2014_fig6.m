%====================================================================
% Linear inverse problem parameter estimate using different weights.
%
% Implementation that generates Figure 6 in the article:
% "Outlier removal for improved source estimation
%  in atmospheric inverse problems",
%  
%  By M. Martinez-Camara, A. Stohl and M. Vetterli, ICASSP 2014
%
%  Marta Martinez-Camara, LCAV, EPFL, 2013
%====================================================================
% Requirements: cvx
close all
clear 

% suppress output from cvx
cvx_quiet(true); 

% Model using mateo dataset 1
load 'Data/matrixGFSXe.mat'
M1 = 1e16*matrix(:,1:120); 

% Model using meteo dataset 2
load 'Data/dataXeECMWF.mat'
M2 = 1e16*M(1:858,1:120); 
[m,n] = size(M1);
load 'Data/measXe.mat'

% scaling measurements
y = 1e16*measurements; 
norm_to_GBqs = 3*60*60 * 1e9;

% where are the differences between both models
Differences = abs(M1 - M2);

% accumulation of errors for each row
Drows = sum((Differences),2);

% exponential weights
ws = exp(-(Drows));
W = diag(ws); 

% Solve the Tikhonov problem with the weights

nLmbds = 20;
lambdas = logspace (3,5,nLmbds);

% selected reg. parameter
lambda = lambdas(17);

% solve the inverse problem
xW = solveInverse(W*M1,W*y,lambda,'NNCTIK');

% plot results
t = 0:3:(3*length(120)/3-1);
figure;
subplot(3,1,1);
plot( xW1(3:3:end)'/norm_to_GBqs, 'b');
title('Fig. 4 : Reconstruction of emission rates in [GBq/s] using the proposed algorithm.');
ylabel('300m-1000m height');

subplot(3,1,2);
plot( xW1(2:3:end)'/norm_to_GBqs, 'y');
ylabel('50m-300m height');

subplot(3,1,3);
plot( xW1(1:3:end)'/norm_to_GBqs, 'r');
ylabel('0m-50m height');
xlabel('time in slices of 3 hours');
