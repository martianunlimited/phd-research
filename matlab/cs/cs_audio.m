% tveq_example.m
%
% Test out tveq code (TV minimization with equality constraints).
%
% Written by: Justin Romberg, Caltech
% Email: jrom@acm.caltech.edu
% Created: October 2005
%

% use implicit, matrix-free algorithms ?  
largescale = 1;

path(path, './Optimization');
path(path, './Measurements');
path(path, './Data');


% test image = 32x32 piece of cameraman's arm
X=y;
largescale=1;
N=ceil(sqrt(size(X,1)))^2;
%X(N)=0;
n = 297;
N = n*n;
%X = phantom(n);
%X = double(imread('../flips/5.1.09.tiff'));

%[sizeX, sizeY]= size(X);
%offsetX=(sizeX-n)/2;
%offsetY=(sizeY-n)/2;
%X=X(offsetX+1:offsetX+n,offsetY+1:offsetY+n);

X = X(1:N);

X = X/norm(X(:));
X = X - mean(X(:));
%x = reshape(X,N,1);
x=X;
figure
% num obs
i=0;
K=[2560 5120 10240 20480 40960 81920]
subplot(2,4,1); plot(X); title(sprintf('Original'));
xlim([0,size(X,1)]);
kCount=size(K,2);
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
if (largescale)
  A = @(z) A_c(z, OMEGA,P);
  At = @(z) At_c(z, N, OMEGA,P);
  % obsevations
  b = A(x);
  % initial point
  x0 = At(b);
else
  FT = 1/sqrt(N)*fft(eye(N));
  A = sqrt(2)*[real(FT(OMEGA,:)); imag(FT(OMEGA,:))];
  A = [1/sqrt(N)*ones(1,N); A];
  At = [];
  % observations
  b = A*x;
  % initial point
  x0 = A'*b;
end

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

                                                                                                                                     
