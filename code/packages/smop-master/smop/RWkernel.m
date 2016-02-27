function [K,runtime]=RWkernel(Graphs, lmbd)
% Compute a random walk kernel for a set of graphs
% Copyright 2011 Karsten Borgwardt, Nino Shervashidze
% Input: Graphs - a 1xN array of graphs represented just with adjacency matrices
% 	 Graphs(i).am is the i'th adjacency matrix
%        Graphs(i) may have other fields, but they will not be considered by this script
%        lmbd - scalar<1: a rule of thumb for setting it is to
%                 take the largest power of 10 which is smaller than 1/d^2,
%                 d being the largest degree in the dataset
% Output: K - NxN kernel matrix K
%         runtime - scalar: runtime in seconds

K=[]; runtime=0;

N=size(Graphs,2);
K=zeros(N,N);

t=cputime; % for measuring runtime

for i=1:N
  for j=i:N
    K(i,j)=K(i,j)+randomwalk(Graphs(i),Graphs(j),lmbd);
    K(j,i)=K(i,j);
  end
end
runtime=cputime-t;
disp(['kernel computation took ', num2str(cputime-t), ' sec']);
end

function result = randomwalk(g1,g2,lmbd)

am1 = g1;
am2 = g2;

[x,rubbish] = pcg(@(x)smtfilter(x,am1,am2,lmbd),ones(size(g1.am,1)*size(g2.am,1),1),1e-6,20);
result =  sum(sum(x));

end

function result = smtfilter(x,am1,am2,lmbd)

yy = vec(am1.am * invvec(x,size(am1.am,1),size(am2.am,1)) * am2.am);

yy = lmbd * yy;

vecu = minus(x,yy);

result = vecu;
end

function v = vec(M)
[m,n] = size(M);
v = reshape(M,m*n,1);
end

function v = invvec(M,m,n)
v = reshape(M,m,n);

end
