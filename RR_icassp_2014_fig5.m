%====================================================================
% Linear inverse problem parameter estimate using different weights.
%
% Implementation that generates Figure 5 in the article:
% "Outlier removal for improved source estimation
%  in atmospheric inverse problems",
%  
%  By M. Martinez-Camara, A. Stohl and M. Vetterli, ICASSP 2014
%
%  Marta Martinez-Camara, LCAV, EPFL, 2013
%====================================================================
% Requirements: cvx

% suppress output from cvx
cvx_quiet(true);

% load ETEX data
load ('./Data/etex_dataset.mat')

% Model using mateo dataset 1
M1 = 1e13*M_era40; 
% Model using meteo dataset 2
M2 = 1e13*M_eraINTERIM; 
[m,n] = size(M1);

% scaling measurements to avoid numerical errors
y = 1e13*measurements;

% where are the differences between both models
Differences = M1 - M2; 
% accumulation of errors for each row
Drows = sum(abs(Differences),2); 

% Build thresholding weights
% we just take the rows with no very big errors
th = 3.5*mean(Drows); 
% which are the indexed of good measurements
goodInx = Drows < th; 
% build linear weights
ws = (1/(Drows + 1).^2)';
% apply threshold
ws(goodInx) = 1;
W_th = diag(ws); 

% Build exponential weights
ws = exp(-(Drows));
W_exp = diag(ws); 

% Define reg. parameter range to test
nLmbds = 70;
lambdas = logspace (-1,3,nLmbds);

% Solve the Tikhonov problem with the th. weights
x_th = zeros(n,nLmbds);
for i = 1:nLmbds
  x_th(:,i) = solveInverse(W_th*M1,W_th*y,lambdas(i),'NNCTIK');
end
mse_th = mean((x_th - repmat(x,1,nLmbds)).^2,1);
[e_th,idx_th] = min(mse_th);


% Solve the Tikhonov problem with the exp. weights
x_exp = zeros(n,nLmbds);
for i = 1:nLmbds
  x_exp(:,i) = solveInverse(W_exp*M1,W_exp*y,lambdas(i),'NNCTIK');
end
mse_exp = mean((x_exp - repmat(x,1,nLmbds)).^2,1);
[e_exp,idx_exp] = min(mse_exp);


% Solve the Tikhonov problem without weights
x_noweights = zeros(n,nLmbds);
for i = 1:nLmbds
  x_noweights(:,i) = solveInverse(M1,y,lambdas(i),'NNCTIK');
end
mse_nw = mean((x_noweights - repmat(x,1,nLmbds)).^2,1);
[e_nw,idx_nw] = min(mse_nw);

% --- plot the results ----------------------------------------------
% nice big figure
figure('position',[100,100,900,500]); 
% th. weights
semilogx(lambdas,mse_th,'k--'); 
hold on
% exp. weights
semilogx(lambdas,mse_exp,'k-'); 
% no weights
semilogx(lambdas,mse_nw,'k-.');
% use large fonts
set(gca,'fontsize',14); 
% easier to read off values
grid on; 
% label the x axis
xlabel('Regularization parameter \beta'); 
% label the y axis
ylabel('Mean square error of reconstruction');
% show the useful range
ylim([20,30]); 
% label the curves
legend('th. weights','exp. weights','without weights','location','northeast'); 




