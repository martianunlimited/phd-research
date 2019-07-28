d=1000;
angle=80;
testSize=1000;
relevantCor=floor(d/2);

rotation=[orth(randn(relevantCor)) zeros(relevantCor,d-relevantCor); zeros(d-relevantCor,relevantCor) eye(d-relevantCor)];
wIdeal=[1 zeros(1,d-1)];
%wIdeal=[0.8 0.6/sqrt(d-1)*ones(1,d-1)];
%wIdeal=[0.5 sqrt(0.75)/sqrt(d-1)*ones(1,d-1)];
wRot=wIdeal*rotation;

tData0=[cos(pi-angle/180*pi)*ones(testSize,1) randnormalized(testSize,d-1)*sin(angle/180*pi)];
tData1=[cos(angle/180*pi)*ones(testSize,1) randnormalized(testSize,d-1)*sin(angle/180*pi)];
    

tData=[tData0; tData1]*rotation;

ensSize=200;
tempRec=zeros(2000,ensSize);
tempEns=zeros(2000,ensSize);
subspaces=[2 5 10 20 50];
subspaceCount=size(subspaces,2);
runs=30;
avgAcc=zeros(runs,subspaceCount,ensSize);
avgFP=zeros(runs,subspaceCount);
modelBinn=zeros(subspaceCount,ensSize);

for runNo=1:runs
for subspaceNo=1:subspaceCount
    k=subspaces(subspaceNo);
for i=1:ensSize
subspace=randperm(d,k);
tempRec(:,i)=tData(:,subspace)*wRot(subspace)';
end
accInd=[tempRec(1:1000,:)<0;tempRec(1001:2000,:)>0];
accInd=single(accInd);


tempRec2=(tempRec(:,:)>0)*2-1;
for j=1:ensSize
tempEns(:,j)=mean(tempRec2(:,1:j),2);
end

accTemp=[tempEns(1:1000,:)<0;tempEns(1001:2000,:)>0];
accTemp=single(accTemp);
accTemp(tempEns==0)=0.5;
avgAcc(runNo,subspaceNo,:)=mean(accTemp,1);
avgFP(runNo,subspaceNo)=1-mean(mean(accInd,1));
avgZero(runNo,subspaceNo)=mean(mean(tempRec==0));
end
end
meanAcc=reshape(mean(avgAcc,1),subspaceCount,ensSize);
meanFP=mean(avgFP);
meanZero=mean(avgZero);
meanFP2=0.5*ones(1,subspaceCount);
meanFP2(meanFP<0.5)=meanFP(meanFP<0.5);%-1/8*meanZero(meanFP<0.5);
temp=((tData*wRot')>0);
accIdeal=sum(temp(1001:2000))/2000 + (1-sum(temp(1:1000)/1000))/2;
%meanFP=meanFP-0.01;

for j=1:subspaceCount
for i=1:ensSize
modelBinn(j,i)=1-cummBinnProb(i,meanFP2(j));
end
end

figure
plot(meanAcc','marker','x','markerSize',1.5)
hold on
plot(modelBinn'-(1-accIdeal),'--', 'lineWidth',2)

