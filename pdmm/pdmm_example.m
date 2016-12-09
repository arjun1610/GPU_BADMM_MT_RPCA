%% By Huahua Wang, the university of Minnesota, Twin cities
%% For details, refer "Parallel Direction Method of Multipliers", NIPS, 2014
%% for the purpose of evaluation of PDMM
%% part of code from http://stanford.edu/~boyd/papers/prox_algs/matrix_decomp.html


m = 1000;
n = 5000;

s = RandStream.create('mt19937ar','seed',5489);
RandStream.setGlobalStream(s);

N = 3;
r = 10;

L = randn(m,r) * randn(r,n);    % low rank
S = sprandn(m,n,0.05);          % sparse
S(S ~= 0) = 20*binornd(1,0.5,nnz(S),1)-10;
V = 0.01*randn(m,n);            % noise

%A = S + L + V;
A =  XO(:, 1:100);

g2_max = norm(A(:),inf);
g3_max = norm(A);
g2 = 0.15*g2_max;
g3 = 0.15*g3_max;

% K = 1;
% h1 = pdmm(A,K,g2,g3);
% pos1 = length(h1.r_norm);
% for k = 1:pos1
%     X_1 = h1.x{k}(:,1:n);
%     X_2 = h1.x{k}(:,n+1:2*n);
%     X_3 = h1.x{k}(:,2*n+1:3*n);
%     h1.objval(k)   = norm(X_1,'fro').^2 + g2*norm(X_2(:),1) + g3*norm(svd(X_3),1);
% end
% 
% K = 2;
% h2 = pdmm(A,K,g2,g3);
% pos2 = length(h2.r_norm);
% for k = 1:pos2
%     X_1 = h2.x{k}(:,1:n);
%     X_2 = h2.x{k}(:,n+1:2*n);
%     X_3 = h2.x{k}(:,2*n+1:3*n);
%     h2.objval(k)   = norm(X_1,'fro').^2 + g2*norm(X_2(:),1) + g3*norm(svd(X_3),1);
% end

K = 3;
h3 = pdmm(A,K,g2,g3);
n = 100;
pos3 = length(h3.r_norm);
for k = 1:pos3
    X_1 = h3.x{k}(:,1:n);
    X_2 = h3.x{k}(:,n+1:2*n);
    X_3 = h3.x{k}(:,2*n+1:3*n);
    h3.objval(k)   = norm(X_1,'fro').^2 + g2*norm(X_2(:),1) + g3*norm(svd(X_3),1);
end

% pos1 = length(h1.r_norm);
% pos2 = length(h2.r_norm);
pos3 = length(h3.r_norm);

figure;
% Create axes
axes1 = axes('FontWeight','bold','FontSize',16);
% Create xlabel
xlabel('time (s)','FontWeight','bold','FontSize',20);
ylabel('residual (log)','FontWeight','bold','FontSize',20);
box(axes1,'on');
hold(axes1,'all');
%plot(h1.time(1:pos1),log10(h1.r_norm+h1.s_norm),'r','LineWidth',2,'DisplayName','PDMM1');
%plot(h2.time(1:pos2),log10(h2.r_norm+h2.s_norm),'g','LineWidth',2,'DisplayName','PDMM2');
plot(h3.time(1:pos3),log10(h3.r_norm+h3.s_norm),'b','LineWidth',2,'DisplayName','PDMM3');
legend('show');

figure;
axes2 = axes('FontWeight','bold','FontSize',16);
% Create xlabel
xlabel('iterations','FontWeight','bold','FontSize',20);
ylabel('residual (log)','FontWeight','bold','FontSize',20);
box(axes2,'on');
hold(axes2,'all');
%plot(1:pos1,log10(h1.r_norm+h1.s_norm),'r','LineWidth',2,'DisplayName','PDMM1');
%plot(1:pos2,log10(h2.r_norm+h2.s_norm),'g','LineWidth',2,'DisplayName','PDMM2');
plot(1:pos3,log10(h3.r_norm+h3.s_norm),'b','LineWidth',2,'DisplayName','PDMM3');
legend('show');

figure;
axes3 = axes('FontWeight','bold','FontSize',16);
% Create xlabel
xlabel('time (s)','FontWeight','bold','FontSize',20);
ylabel('objective (log)','FontWeight','bold','FontSize',20);
box(axes3,'on');
hold(axes3,'all');
%plot(h1.time(1:pos1),log10(h1.objval(1:pos1)),'r','LineWidth',2,'DisplayName','PDMM1');
%plot(h2.time(1:pos2),log10(h2.objval(1:pos2)),'g','LineWidth',2,'DisplayName','PDMM2');
plot(h3.time(1:pos3),log10(h3.objval(1:pos3)),'b','LineWidth',2,'DisplayName','PDMM3');
legend('show');

figure;
axes4 = axes('FontWeight','bold','FontSize',16);
% Create xlabel
xlabel('iterations','FontWeight','bold','FontSize',20);
ylabel('objective (log)','FontWeight','bold','FontSize',20);
box(axes4,'on');
hold(axes4,'all');
% plot(1:pos1,log10(h1.objval(1:pos1)),'r','LineWidth',2,'DisplayName','PDMM1');
% plot(1:pos2,log10(h2.objval(1:pos2)),'g','LineWidth',2,'DisplayName','PDMM2');
plot(1:pos3,log10(h3.objval(1:pos3)),'b','LineWidth',2,'DisplayName','PDMM3');
legend('show');