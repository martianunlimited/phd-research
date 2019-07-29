% tveq_example.m
%
% TV minimization with equality constraints).
%
% Adapted from example by Justin Romberg, Caltech
% Email: jrom@acm.caltech.edu
% Created: October 2005
%

% use implicit, matrix-free algorithms ?  
files=dir('../dataset/audio/*.ogg');
fileCount=size(files,1);
n = 297;
N = n*n;
K=[2560 5120 10240 20480 40960 81920];
kCount=size(K,2);


for fileNo=1:fileCount
% test image = 32x32 piece of cameraman's arm
filename=files(fileNo).name;
[X fs]=audioread(['../dataset/audio/' filename]);

start=randi(size(X,1)-N);
X = X(start+1:start+N);

X = X/norm(X(:));
X = X - mean(X(:));

x=X;
figure

i=0;

subplot(2,4,1); plot(X); title(sprintf('Original'));
xlim([0,size(X,1)]);

reconsX=zeros(kCount,N);
i=0;
for k=K
i=i+1;
% num obs

% permutation P and observation set OMEGA
P = randperm(N)';
%P=1:N;
q = randperm(N,k);
OMEGA = q;



% measurement matrix
A = @(z) A_c(z, OMEGA,P);
At = @(z) At_c(z, N, OMEGA,P);
% obsevations
b = A(X);
% initial point
x0 = At(b);

%Xbp=reshape(x0,n,n);
epsilon = 5e-3;
%tvI = sum(sum(sqrt([diff(X,1,2) zeros(n,1)].^2 + [diff(X,1,1); zeros(1,n)].^2 )));
%disp(sprintf('Original TV = %.3f', tvI));

time0 = clock;
%xp =  l1eq_logbarrier(x0, A, At, b, 1e-3, 1e-3, 10, 1e-8, 200);
%xp = tvqc_logbarrier(x0,A,At,b);
xp =  tvqc_logbarrier(x0, A, At, b, epsilon, 1e-3, 5, 1e-8, 200);
xp =  tveq_logbarrier(x0, A, At, b, 1e-3, 10, 1e-8, 200);

%Xp = reshape(xp, n, n);
%disp(sprintf('Total elapsed time = %f secs\n', etime(clock,time0)));
xp=xp/norm(xp);
%i=i+1;
subplot(2,4,i+2); plot(real(xp)); title(sprintf('Subspace=%d',k));
xlim([0,size(X,1)]);
reconsX(i,:)=xp;
RSSres(i)=sum((X-xp).^2);
end
end
                                                                                                                                     
