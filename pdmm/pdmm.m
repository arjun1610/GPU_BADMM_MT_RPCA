%% By Huahua Wang, the university of Minnesota, Twin cities
%% For details, refer "Parallel Direction Method of Multipliers", NIPS, 2014
%% for the purpose of evaluation of PDMM
%% part of code from http://stanford.edu/~boyd/papers/prox_algs/matrix_decomp.html

 
function h = pdmm(A,K,g2,g3)

[m,n] = size(A);
N = 3;

MAX_ITER = 200;

PRINT = 0;  %% print intermediate result

tic;

%% intialization
lambda = 1;
rho = 1/lambda;

X1 = zeros(m,n);
X2 = zeros(m,n);
X3 = zeros(m,n);
U   = zeros(m,n);       %% scaled dual: Y/rho
B = -A;                 %% Yhat/rho+(X1+X2+X3-A)
x = 0;
idx = zeros(K,1);

% default value of step size
Ki = min(K,N);
tau = K/(Ki*(2*N-K));
nu = 1/Ki;                %% 1-nu in the paper

if PRINT 
    fprintf('\n%3s\t%10s\t%10s\n', 'iter','r norm', 'objective');
end

for k = 1:MAX_ITER

    %% x-update
    %%%% random
    % idx = randperm(N,K);
    switch K
        case 1
            idx = mod(k,N)+1;
        case 2
            idx(1) = mod(idx(2),3) + 1;
            idx(2) = mod(idx(1),3) + 1;
        case 3
            idx = 1:N;
    end

    for i = 1:K
        switch idx(i)
            case 1
                X1 = (1/(1+lambda))*(X1 - B);
            case 2
                X2 = prox_l1(X2 - B, lambda*g2);
            case 3
                X3 = prox_matrix(X3 - B, lambda*g3, @prox_l1);
        end
    end
    
    xold = x;
    x = [X1 X2 X3];

    % u-update
    Uold = U;
    R = (X1 + X2 + X3 - A);
    U = U + R*tau;
    B = U + R*nu;

    h.r_norm(k)   = norm( x - xold );
    h.s_norm(k) = norm( U - Uold );
    h.x{k} = x;
    
    h.time(k) = toc;

     if PRINT && (k == 1 || mod(k,10) == 0)
         h.objval(k)   = objective(X_1, g2, X_2, g3, X_3);
         fprintf('%4d\t%10.4f\t%10.2f\n', k, h.r_norm(k)+h.s_norm(k), h.objval(k));
     end

    if h.r_norm(k)+h.s_norm(k) < 1e-4 && k > 5
         break;
    end

end

end

%% code from http://stanford.edu/~boyd/papers/prox_algs/matrix_decomp.html
function x = avg(varargin)
    N = length(varargin);
    x = 0;
    for k = 1:N
        x = x + varargin{k};
    end
    x = x/N;
end

function p = objective(X_1, g_2, X_2, g_3, X_3)
    p = norm(X_1,'fro').^2 + g_2*norm(X_2(:),1) + g_3*norm(svd(X_3),1);
end

function x = prox_l1(v, lambda)
% PROX_L1    The proximal operator of the l1 norm.
%
%   prox_l1(v,lambda) is the proximal operator of the l1 norm
%   with parameter lambda.

    x = max(0, v - lambda) - max(0, -v - lambda);
end

function x = prox_matrix(v, lambda, prox_f)
% PROX_MATRIX    The proximal operator of a matrix function.

    [U,S,V] = svd(v,'econ');
    x = U*diag(prox_f(diag(S), lambda))*V';
end