function x=solveInverse(A,b,params,type)
%====================================================================
% A generic function to solve inverse problems Ax=b
% 
%    A - sensing matrix (n-by-m)
%    b - measurements vector (n-by-1)
%    params - optimization parameters (lambdas, etc.)
%    type - type of solution to use
%            LS - least squares
%            TIK - Tikhonov regularization
%            LASSO - l1 regularization
%            NNCLS - non-negative constraint least squares
%            NNCTIK - non-negative constraint Tikhonov
%            NNCL1 - non-negative constrained sparse
%     
%    x - solution vector (m-by-1)
% 
% Marta Martinez-Camara
% LCAV, EPFL
%====================================================================
[m,n]=size(A); % discover m and n
x=zeros(n,1); % empty container for the solution
switch type % different algorithms we may want to try
%--------------------------------------------------------------------
  case 'LS' % least squares
    % simple minimization of ||Ax-b||2
    cvx_begin; % start cvx 
      variable x(n); % declare the variable cvx will optimize
      minimize((A*x-b)'*(A*x-b)); % declare the cost function
    cvx_end; % and that was it; end the problem specification
%--------------------------------------------------------------------
  case 'NNLS' % non-negative constrained least squares
    % same as LS, but constrained to non-negative elements in x
    cvx_begin; % start cvx 
      variable x(n); % declare the variable cvx will optimize
      minimize((A*x-b)'*(A*x-b)); % declare the cost function
      subject to % here come the constraints
      x>=0 ; % non-negative contraint
    cvx_end; % and that was it; end the problem specification
%--------------------------------------------------------------------
  case 'NNCTIK' % non-negative constrained Tikhonov
    % LS + minimum energy + non-negative constraint
    cvx_begin; % start cvx 
      variable x(n); % declare the variable cvx will optimize
      minimize((A*x-b)'*(A*x-b)+params(1)*norm(x,2)); % declare the cost function
      subject to % here come the constraints
      x >= 0 ; % non-negative contraint
    cvx_end; % and that was it; end the problem specification
%--------------------------------------------------------------------
  case 'NNCL1' % non-negative constrained l1
    % LS + sparse + non-negative constraint
    %nCol = sqrt(sum(A.^2,1));
    %W = diag(nCol);
    cvx_begin; % start cvx 
      variable x(n); % declare the variable cvx will optimize
      minimize((A*x-b)'*(A*x-b)+params(1)*norm(x,1)); % declare the cost function
      subject to % here come the constraints
      x >= 0 ; % non-negative contraint
    cvx_end; % and that was it; end the problem specification
  
%--------------------------------------------------------------------
  case 'TIK' % Tikhonov
    % LS + minimum energy + non-negative constraint
    cvx_begin; % start cvx 
      variable x(n); % declare the variable cvx will optimize
      minimize((A*x-b)'*(A*x-b)+params(1)*norm(x,2)); % declare the cost function
    cvx_end; % and that was it; end the problem specification
%--------------------------------------------------------------------
  case 'LASSO' %lasso
    cvx_begin; % start cvx 
      variable x(n); % declare the variable cvx will optimize
      minimize((A*x-b)'*(A*x-b)+params(1)*norm(x,1)); % declare the cost function
    cvx_end; % and that was it; end the problem specification    

  otherwise % an unknown method!!
    error(['unknown method: ',type]); % exit with an error, best we can do...
end % switch statement
end % function
%EOF