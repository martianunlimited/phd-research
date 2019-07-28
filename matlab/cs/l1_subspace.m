% Run 1000 simulation of CS recovery. Never plot all 1000 runs, unless you want 
% a really bad time with your computer
run=10000;
plotCount=8; 
failureCounter=zeros(1,3);

for i=1:run
% Setup of the problem, we have T=5 \pm 1 signals, with dimensionality N =
% 200, and, +ve signals are more likely than -ve signals
T=5;
N=200;
k=25;
x=zeros(N,1);
x(randperm(N,T))=sign(randn(T,1)+0.4);
R=orth(randn(N,k))';

% L1-Magic requires a positive definite matrix A for compressive sensing, we 
% achieve this by reflecting the RS matrix, 
% i) randomly 
% ii) based on the signal we want to recover this also has the added benefit 
% of ``densifying'' the problem and requiring less compressive samples. 
% C'=sqrt(N/T/2) instead of sqrt(N/T), see results on flipping probability 

%taking \epsilon away from two random \pm 1 matrices to avoid singular
%matrices
%hn=householderNormal(0.0001-householder_sv(N),householder_sv(N));
hn=householderNormal(x,householder_sv(N));
h=eye(N)-2*hn*hn';
P=randsubspace(k,N);
PH=P*h;

% We run L1-Magic and let it do it's ``magic'' 
yRP=R*x;
x0RP=R'*yRP;
yRS=PH*x;
x0RS=PH'*yRS;
xpRP=l1eq_pd(x0RP,R,[],yRP,1e-4);
xpRS=l1eq_pd(x0RS,PH,[],yRS,1e-4);

% We plot the results, note that the recovery may fail, ~42% of the time
% for RP and RS fails almost 100% of the time. 
if (i<(plotCount+1))
    figure
    subplot(4,1,1);
    plot(x);
    title('Original Signal');
    ylim([-1 1]);
    subplot(4,1,2);
    plot(xpRP);
    titleStr=sprintf('Recovery with RP k=%d',k);
    title(titleStr);
    ylim([-1 1]);
    subplot(4,1,3);
    plot(xpRS);
    titleStr=sprintf('Recovery with RS k=%d',k);
    title(titleStr);
    ylim([-1 1]);
end 

% We increase the number of compressive samples for RS by a factor of c' (and take the
% next integer) RS now fails, ~45% of the time
k=ceil(k*sqrt(N/T/2));
P2=randsubspace(k,N);
PH2=P2*h;
yRS2=PH2*x;
x0RS2=PH2'*yRS2;
xpRS2=l1eq_pd(x0RS2,PH2,[],yRS2,1e-4);

% Count the number of failures, a failure is missing a signal.. note that
% the zeros values are not truly zero, and we need to increase the
% threshold to avoid false positives
threshold=0.9;
if (sum(abs(x-xpRS2))>threshold) 
failureCounter(3)=failureCounter(3)+1;
end
if (sum(abs(x-xpRS))>threshold) 
failureCounter(2)=failureCounter(2)+1;
end
if (sum(abs(x-xpRP))>threshold) 
failureCounter(1)=failureCounter(1)+1;
end

if(i<(plotCount+1))
    subplot(4,1,4);
    plot(xpRS2);
    titleStr=sprintf('Recovery with Rs k=%d',k);
    title(titleStr);
    ylim([-1 1]);
end
end