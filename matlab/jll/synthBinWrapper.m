prob=[0.0001 0.002 0.02 0.035 0.05 0.10 0.15 0.25 0.5];
dim=100000;

probCount=size(prob,2);
subspaces=[5 7 10 15 25 35 50 70 100 150 250 350 500 700 1000 1500 2500 3500 5000 7000 10000 15000 25000 35000 50000 70000 100000];
subspaceCount=size(subspaces,2);
obs=10;

epsilons=[0.05 0.1 0.2 0.5 1];
epsilonCount=size(epsilons,2);
subspaceStat=zeros(probCount,subspaceCount,4);
hsubspaceStat=zeros(probCount,subspaceCount,4);

subspaceInt=zeros(probCount,subspaceCount,3);
hsubspaceInt=zeros(probCount,subspaceCount,3);
fileNo=0;

normSub=zeros(probCount,subspaceCount,obs*(obs-1)/2);
normHSub=zeros(probCount,subspaceCount,obs*(obs-1)/2);

subspaceProb=zeros(probCount,subspaceCount,epsilonCount);
hsubspaceProb=zeros(probCount,subspaceCount,epsilonCount);

for probNo=1:probCount
    baseProb=prob(probNo)
    dim=100000;
        

  
        
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
l=0;
tic;
    
    P=randperm(dim,ceil(subspace));
    P=sort(P);
    SP=S(P,:);
for i=1:obs
    for j=i+1:obs
l=l+1;
        u=S(:,i);
        v=S(:,j);
        uv=u-v;
        uP=SP(:,i);
        vP=SP(:,j);
        uvP=uP-vP;
        normSub(probNo,k,l)=sum(uvP.^2) / sum(uv.^2)*dim/ceil(subspace);
    end
end





subspaceStat(probNo,k,:)=[mean(normSub(probNo,k,:)) var(normSub(probNo,k,:)) min(normSub(probNo,k,:)) max(normSub(probNo,k,:))];
hsubspaceStat(probNo,k,:)=[mean(normHSub(probNo,k,:)) var(normHSub(probNo,k,:)) min(normHSub(probNo,k,:)) max(normHSub(probNo,k,:))];

subspaceInt(probNo,k,:)=prctile(normSub(probNo,k,:),[5 50 95]);
hsubspaceInt(probNo,k,:)=prctile(normHSub(probNo,k,:),[5 50 95]);
for l=1:epsilonCount
    epsilon=epsilons(l);
    subspaceProb(probNo,k,l)=(sum((abs(normSub(probNo,k,:)-1))>=epsilon)/obs);
end


    end


end

bern=zeros(probCount,subspaceCount,epsilonCount);
hoff=zeros(probCount,subspaceCount,epsilonCount);
test=zeros(probCount,subspaceCount,epsilonCount);
test2=zeros(probCount,subspaceCount,epsilonCount);

figure;
plotCount=0
for i=1:probCount
    for j=1:epsilonCount
        plotCount=plotCount+1
        % Bennett bound that was abandoned but code kept in case we want to
        % revisit it
        s=subspaces/dim;
        tau=(1-2.*(s))./(4.*sqrt(s).*log(1./s - 1));
        u=epsilons(j) / (1-prob(i));
        
        % Bernstein's bound
        % we use the greater of the two tails (exponential tail or gaussian tail, 
        % but epsilon is should be too small to result in an exponential tail, this is here for completeness 
        % note that c' and c has an easy form for binomial distributions.
        % (it's simply 1/proportions of 1's in the data) 
        bernmult=min(subspaces*epsilons(j)^2,sqrt(subspaces*dim)*epsilons(j));
        bern(i,:,j)=min(2*exp(-prob(i)*bernmult/2),1);
        
        % Hoeffding's Bounds
        hoff(i,:,j)=min(2*exp(-prob(i)^2*epsilons(j)^2*subspaces),1);
        subplot(probCount,epsilonCount,plotCount)
        semilogx(subspaces,subspaceProb(i,:,j));
        hold on;
        semilogx(subspaces,bern(i,:,j));
        semilogx(subspaces,hoff(i,:,j));
        xlim([5 100000]);
        ylim([0.0001 1])
    end
end
