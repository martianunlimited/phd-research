files=dir('../dataset/images/*.tiff');
fileCount=size(files,1);

n = 256;
N = n*n;
K=[25 100 500 2500 10000 50000];
sizeK=size(K,2);
mseStat=zeros(fileCount,sizeK,3);
dctStat=zeros(fileCount,2);
runTime=zeros(fileCount,sizeK,3);
for fileNo=1:1:fileCount;
    filename=files(fileNo).name
    X = double(imread(['../dataset/images/' filename]));
[sizeX, sizeY]= size(X);
offsetX=(sizeX-n)/2;
offsetY=(sizeY-n)/2;

X=X(offsetX+1:offsetX+n,offsetY+1:offsetY+n);

x = X(:);

X = X/norm(X(:));
X = X - mean(X(:));
x = reshape(X,N,1);
P = randperm(N)';
P1=1:N;
figure

i=0;
temp=dct(x(P));
cConst=mean(temp.^4)/(mean(temp.^2)^2);
temp=dct(x(P1));
cConst1=mean(temp.^4)/(mean(temp.^2)^2);
dctStat(fileNo,1)=cConst;
dctStat(fileNo,2)=cConst1;

for kInd=1:sizeK
k=K(kInd)
% permutation P and observation set OMEGA
q = randperm(N,k);
OMEGA = q';
if k<5000 
    rp=randn(k,N);
end



% measurement matrix

  A = @(z) A_cg(z, rp,P);
  At = @(z) At_cg(z, N, rp,P);
  As = @(z) A_c(z, OMEGA,P);
  Ast = @(z) At_c(z, N, OMEGA,P);
  As1 = @(z) A_c(z, OMEGA,P1);
  Ast1 = @(z) At_c(z, N, OMEGA,P1);
  % obsevations
  b = A(x);
  bs = As(x);
  bs1 = As1(x);
  
  % initial point
  x0 = At(b);
  xs0 = Ast(bs);
  xs10 = Ast1(bs);

Xbp=reshape(x0,n,n);
epsilon = 5e-3;


tvI = sum(sum(sqrt([diff(X,1,2) zeros(n,1)].^2 + [diff(X,1,1); zeros(1,n)].^2 )));
disp(sprintf('Original TV = %.3f', tvI));
if k<5000
    time0 = clock;
    xp =  tvqc_logbarrier(x0, A, At, b, epsilon, 1e-3, 5, 1e-8, 200);
    disp(sprintf('Total elapsed time = %f secs\n', etime(clock,time0)));
    runTime(fileNo,kInd,1)=etime(clock,time0);
end
time0 = clock;
xps =  tvqc_logbarrier(xs0, As, Ast, bs, epsilon, 1e-3, 5, 1e-8, 200);
disp(sprintf('Total elapsed time = %f secs\n', etime(clock,time0)));
    runTime(fileNo,kInd,2)=etime(clock,time0);
time0 = clock;
xps1 =  tvqc_logbarrier(xs10, As1, Ast1, bs1, epsilon, 1e-3, 5, 1e-8, 200);
disp(sprintf('Total elapsed time = %f secs\n', etime(clock,time0)));
    runTime(fileNo,kInd,3)=etime(clock,time0);
Xp = reshape(xp, n, n);
Xps = reshape(xps, n, n);
Xps1 = reshape(xps1, n, n);

mseStat(fileNo,kInd,1)=sqrt(mean((x-xp).^2))/sqrt(mean(x.^2));
mseStat(fileNo,kInd,2)=sqrt(mean((x-xps).^2))/sqrt(mean(x.^2));
mseStat(fileNo,kInd,3)=sqrt(mean((x-xps1).^2))/sqrt(mean(x.^2));

i=i+1;
if k<5000
    subplot(3,sizeK,i); imagesc(Xp);colormap(gray); title(sprintf('Gaussian RP k= %d',k))
end
subplot(3,sizeK,i+sizeK); imagesc(Xps);colormap(gray); title(sprintf('Dense RS k=%d',k));
subplot(3,sizeK,i+(2*sizeK)); imagesc(Xps1);colormap(gray);title(sprintf('Sparse RS k=%d',k))
end
subplot(3,sizeK,sizeK); imagesc(X);colormap(gray); title('Reference Image');
end