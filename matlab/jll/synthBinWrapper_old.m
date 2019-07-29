%% Previous Wrapper to generate the results for figure 4.2, this code have been superceded

prob=[0.01 0.02 0.035 0.05 0.10 0.15 0.25 0.5];
dim=10000;

probCount=size(prob,2);
subspaces=[5 7 10 15 25 35 50 70 100 150 250 350 500 700 1000 1500 2500 3500 5000 7000 10000];
subspaceCount=size(subspaces,2);
pairs=100;
obs=4000;
epsilons=[0.01 0.02 0.03 0.05 0.07 0.1 0.15 0.2 0.3 0.5 0.7 1];
epsilonCount=size(epsilons,2);
subspaceStat=zeros(probCount,subspaceCount,4);
hsubspaceStat=zeros(probCount,subspaceCount,4);

subspaceInt=zeros(probCount,subspaceCount,3);
hsubspaceInt=zeros(probCount,subspaceCount,3);
fileNo=0;

normSub=zeros(probCount,subspaceCount,pairs*(pairs-1)/2);
normHSub=zeros(probCount,subspaceCount,pairs*(pairs-1)/2);

subspaceProb=zeros(probCount,subspaceCount,epsilonCount);
hsubspaceProb=zeros(probCount,subspaceCount,epsilonCount);

for probNo=1:probCount
        baseProb=prob(probNo)
        dim=10000
        

  
        
S=binRandGen(baseProb,dim,obs);

dim=size(S,1);
obs=size(S,2);
tic
mn=mean(S,2);
v=ones(1,dim)*sqrt(1/dim);
v=v/norm(v);

toc
alpha=full(v*S);

scale2=1;
scale=1;

for k=1:size(subspaces,2)    
    subspace=subspaces(k)
    randIdx=randperm(obs,pairs);
l=0;
tic;
    
    P=randperm(dim,ceil(subspace));
    P=sort(P);
    SP=S(P,:);
for i=1:pairs
    for j=i+1:pairs
        l=l+1;
        uv=S(:,randIdx(i))-S(:,randIdx(j));
        uvP=SP(:,randIdx(i))-SP(:,randIdx(j));   
        normSub(probNo,k,l)=sqrt(sum(uvP.^2) / sum(uv.^2)*dim/ceil(subspace));
        uv2=full(S(:,randIdx(i))-S(:,randIdx(j)));
        uvP2=abs(uv(P));
        uvS=sum(abs(uv2));
        uvP2=uvP2-2*uvS/dim;
        normHSub(probNo,k,l)=sqrt((sum(uvP2.^2)) / sum(uv.^2)*dim/ceil(subspace));
    end
end





subspaceStat(probNo,k,:)=[mean(normSub(probNo,k,:)) var(normSub(probNo,k,:)) min(normSub(probNo,k,:)) max(normSub(probNo,k,:))];
hsubspaceStat(probNo,k,:)=[mean(normHSub(probNo,k,:)) var(normHSub(probNo,k,:)) min(normHSub(probNo,k,:)) max(normHSub(probNo,k,:))];

subspaceInt(probNo,k,:)=prctile(normSub(probNo,k,:),[5 50 95]);
hsubspaceInt(probNo,k,:)=prctile(normHSub(probNo,k,:),[5 50 95]);

for l=1:epsilonCount
    epsilon=epsilons(l);
    subspaceProb(probNo,k,l)=1- (sum(abs(normSub(probNo,k,:)-1)<epsilon)/(pairs*(pairs-1)/2));
    hsubspaceProb(probNo,k,l)=1- (sum(abs(normSub(probNo,k,:)-1)<epsilon)/(pairs*(pairs-1)/2));
end


    end


end

bern=zeros(probCount,subspaceCount,epsilonCount);
hoff=zeros(probCount,subspaceCount,epsilonCount);
test=zeros(probCount,subspaceCount,epsilonCount);

probCount=5;
epsilonCount=6;
figure
plotCount=0
%% The hoeffding and bernstein inequality has a nice simple form for Binominal related to the 
% proportions of 1s 
for i=[2 4 5 7 8] %1:probCount
    for j=[4 6 8 10 11 12] %1:epsilonCount
        plotCount=plotCount+1
        bern(i,:,j)=min(2*exp(-prob(i)*epsilons(j)^2*subspaces),1);
        hoff(i,:,j)=min(2*exp(-prob(i)^2*2*epsilons(j)^2*subspaces),1);
        subplot(probCount,epsilonCount,plotCount);

        semilogx(subspaces,subspaceProb(i,:,j),'--r');
        hold on;
        semilogx(subspaces,bern(i,:,j),'-g');
        semilogx(subspaces,hoff(i,:,j),'-.b');
        xlim([5 10000]);
        ylim([0 1]);
    end
end
