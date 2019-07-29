
%Define definition of the run, including range of subspaces, files to run,
%number of pairs of vectors, window size and number of random windows
files=dir('../dataset/images/*.tiff');
fileCount=size(files,1);
runCount=1000;
windSize=50
subspaces=[5 10];% 15 20 25 35 50 60 75 100 125 150 200 250 300 350 400 450 500 550 600 650 750];
subspaceCount=size(subspaces,2);
pairs=100;



%% Initialize all statistics to 0
subspaceInt=zeros(fileCount,subspaceCount,3);
hsubspaceInt=zeros(fileCount,subspaceCount,3);
RPInt=zeros(fileCount,subspaceCount,3);
SRPInt=zeros(fileCount,subspaceCount,3);
PCAInt=zeros(fileCount,subspaceCount,3);
DCTInt=zeros(fileCount,subspaceCount,3);

subspaceStat=zeros(fileCount,subspaceCount,4);
hsubspaceStat=zeros(fileCount,subspaceCount,4);
RPStat=zeros(fileCount,subspaceCount,4);
SRPStat=zeros(fileCount,subspaceCount,4);
PCAStat=zeros(fileCount,subspaceCount,4);
DCTStat=zeros(fileCount,subspaceCount,4);

subspaceRunTime=zeros(fileCount,subspaceCount);
hsubspaceRunTime=zeros(fileCount,subspaceCount);
rpRunTime=zeros(fileCount,subspaceCount);
srpRunTime=zeros(fileCount,subspaceCount);
pcaRunTime=zeros(fileCount,subspaceCount);
PCATime=zeros(fileCount,1);
HPreTime=zeros(fileCount,1);

normPX=zeros(fileCount,subspaceCount,pairs*(pairs-1)/2);
normRX=zeros(fileCount,subspaceCount,pairs*(pairs-1)/2);
normSRX=zeros(fileCount,subspaceCount,pairs*(pairs-1)/2);
normPCA=zeros(fileCount,subspaceCount,pairs*(pairs-1)/2);
%normDCT=zeros(fileCount,subspaceCount,pairs*(pairs-1)/2);
normHPX=zeros(fileCount,subspaceCount,pairs*(pairs-1)/2);

scaleFactor=zeros(fileCount,1);
scaleFactor2=zeros(fileCount,1);
%% Start of the Outer loop 
for fileNo=1:1:fileCount
% Loop through each file and store data as a data matrix
filename=files(fileNo).name
X=imread(['../dataset/images/' filename]);
X=double(X);
sqrt(max(max(X.^2/mean(mean(X.^2)))))

% Randomly generate $runCount windows of size $windSize and store the
% vectors as Xvec

sizeY=size(X,2)
sizeX=size(X,1)
coorXY=zeros(runCount,2);
Xvec=zeros(runCount,windSize^2);

for k=1:runCount
coorXY(k,:)=[randi([1 sizeX-windSize]) randi([1 sizeY-windSize])];
Xvec(k,:)=reshape(X(coorXY(k,1):coorXY(k,1)+windSize-1,coorXY(k,2):coorXY(k,2)+windSize-1),1,[]);
% Experiment 1: Normalize the vectors to have norm 1 
% Xvec(k,:)=Xvec(k,:)/norm(Xvec(k,:));
end
% Experiment 2: Mean centering;
% Xvec=Xvec-mean(mean(Xvec));
%

% Preprocessing for Householder Transforms
tic;
v=randn(1,windSize^2)/sqrt(windSize^2);
alpha=Xvec*v';
HPreTime(fileNo)=toc;

tic
[pcacov, dummy1, dummy2, dummy3, pcaExplained]=pca(Xvec);
PCATime(fileNo)=toc;

%tic
%dctX=dct(Xvec);
%toc
%scale=sum(var(Xvec'.^2))*dim/((sum(sum(Xvec.^2))/obs)^2)/4
scale=sqrt(sum(sum(X.^4))*size(X,1)*size(X,2)/(sum(sum(X.^2)))^2*8);
X1=reshape(X,1,[]);
scale2=(size(X1,2)*(max(X1.^2)-min(X1.^2)))/sum(X1.^2);
scaleFactor(fileNo)=scale;
scaleFactor2(fileNo)=scale2;

% For varying size of the subspace;
for k=1:size(subspaces,2)
subspace=subspaces(k)


% Randomply generate $pairs observations from $runCount observations
randInd=randperm(runCount,pairs);

index=0;

% Random Subspace Run
tic;
P=randperm(windSize^2,subspace);
XP=Xvec(:,P);
for i = 1:pairs
    for j= i+1 : pairs
    index=index+1;
    uv=(Xvec(randInd(i),:)-Xvec(randInd(j),:))';
    uvP=(XP(randInd(i),:)-XP(randInd(j),:))';
    dist=norm(uv);
    if (dist==0) 
        continue;      
    end   
    normPX(fileNo,k,index)=norm(uvP)/dist*sqrt(windSize^2/subspace);
    end
end
subspaceRunTime(fileNo,k)=toc;

index=0;
%Random Subspace with Householder Transfrom
tic;
P=randperm(windSize^2,subspace);
XP=Xvec(:,P);
for i = 1:pairs
    for j= i+1 : pairs
    index=index+1;
    uv=(Xvec(randInd(i),:)-Xvec(randInd(j),:))';
    uvP=(XP(randInd(i),:)-XP(randInd(j),:))';
    alpha2=alpha(randInd(i))-alpha(randInd(j));
    uvP=uvP-2*alpha2*v(P)';
    dist=norm(uv);
    if (dist==0) 
        continue;      
    end   
    normHPX(fileNo,k,index)=norm(uvP)/dist*sqrt(windSize^2/subspace);
    end
end
hsubspaceRunTime(fileNo,k)=toc;



index=0;
% Random Gaussian Projection
tic;
R=randproj(subspace,windSize^2);
XP=Xvec*R';
for i = 1:pairs
    for j= i+1 : pairs
    index=index+1;
    uv=(Xvec(randInd(i),:)-Xvec(randInd(j),:))';
    uvP=(XP(randInd(i),:)-XP(randInd(j),:))';
    dist=norm(uv);
    if (dist==0) 
        continue;      
    end
    
    normRX(fileNo,k,index)=norm(uvP)/dist*sqrt(windSize^2/subspace);
    end
end
rpRunTime(fileNo,k)=toc;

index=0;
% Sparse Random Subspace (Achlioptas Projection)
tic;
SR=sparse(randproj_3(subspace,windSize^2));
XP=Xvec*SR';
for i = 1:pairs
    for j= i+1 : pairs
    index=index+1;
    uv=(Xvec(randInd(i),:)-Xvec(randInd(j),:))'; 
    uvP=(XP(randInd(i),:)-XP(randInd(j),:))';  
    dist=norm(uv);
    if (dist==0) 
        continue;      
    end
    
    normSRX(fileNo,k,index)=norm(uvP)/dist*sqrt(1/subspace);
    end
end
srpRunTime(fileNo,k)=toc;


index=0;
tic;
% Principal Component Analysis
PCAProj=pcacov(:,1:subspace);
XP=Xvec*PCAProj;
for i = 1:pairs
    for j= i+1 : pairs
    index=index+1;
    uv=(Xvec(randInd(i),:)-Xvec(randInd(j),:))';  
    uvP=(XP(randInd(i),:)-XP(randInd(j),:))';     
    dist=norm(uv);
    if (dist==0) 
        continue;      
    end
    
    normPCA(fileNo,k,index)=norm(uvP)/dist/sum(pcaExplained(1:subspace))*sum(pcaExplained);
    end
end
pcaRunTime(fileNo,k)=toc;

%Obtain the statistics for the various runs
subspaceStat(fileNo,k,:)=[mean(normPX(fileNo,k,:)) var(normPX(fileNo,k,:)) min(normPX(fileNo,k,:)) max(normPX(fileNo,k,:))];
RPStat(fileNo,k,:)=[mean(normRX(fileNo,k,:)) var(normRX(fileNo,k,:)) min(normRX(fileNo,k,:)) max(normRX(fileNo,k,:))];
SRPStat(fileNo,k,:)=[mean(normSRX(fileNo,k,:)) var(normSRX(fileNo,k,:)) min(normSRX(fileNo,k,:)) max(normSRX(fileNo,k,:))];
PCAStat(fileNo,k,:)=[mean(normPCA(fileNo,k,:)) var(normPCA(fileNo,k,:)) min(normPCA(fileNo,k,:)) max(normPCA(fileNo,k,:))];
hsubspaceStat(fileNo,k,:)=[mean(normHPX(fileNo,k,:)) var(normHPX(fileNo,k,:)) min(normHPX(fileNo,k,:)) max(normHPX(fileNo,k,:))];

subspaceInt(fileNo,k,:)=prctile(normPX(fileNo,k,:),[5 50 95]);
RPInt(fileNo,k,:)=prctile(normRX(fileNo,k,:),[5 50 95]);
SRPInt(fileNo,k,:)=prctile(normSRX(fileNo,k,:),[5 50 95]);
PCAInt(fileNo,k,:)=prctile(normPCA(fileNo,k,:),[5 50 95]);
hsubspaceInt(fileNo,k,:)=prctile(normHPX(fileNo,k,:),[5 50 95]);

% End of the Sub
end

end



avgRPStat=mean(RPStat,1);
avgSRPStat=mean(SRPStat,1);
avgSubspaceStat=mean(subspaceStat,1);
avgHSubspaceStat=mean(hsubspaceStat,1);
avgPCAStat=mean(PCAStat,1);

avgRPInt=mean(RPInt,1);
avgSRPInt=mean(SRPInt,1);
avgSubspaceInt=mean(subspaceInt,1);
avgHSubspaceInt=mean(hsubspaceInt,1);
avgPCAInt=mean(PCAInt,1);

figure;
subplot(2,2,1);
pr1=plot(subspaces, avgRPInt(1,:,[1]),'r:');
hold on
pr2=plot(subspaces, avgRPStat(1,:,1),'r.:');
pr3=plot(subspaces, avgRPInt(1,:,[3]),'r:');
hold off

subplot(2,2,2);
ps1=plot(subspaces, avgSubspaceInt(1,:,[1]),'b:');
hold on;
ps2=plot(subspaces, avgSubspaceStat(1,:,1),'b+:');
ps3=plot(subspaces, avgSubspaceInt(1,:,[3]),'b:');
hold off;

subplot(2,2,3);
ppc1=plot(subspaces, avgPCAInt(1,:,[1]),'g:');
hold on;
ppc2=plot(subspaces, avgPCAStat(1,:,1),'go:');
ppc3=plot(subspaces, avgPCAInt(1,:,[3]),'g:');
hold off;

subplot(2,2,4);
psr1=plot(subspaces, avgSRPInt(1,:,[1]),'m:');
hold on;
psr2=plot(subspaces, avgSRPStat(1,:,1),'mx:');
psr3=plot(subspaces, avgSRPInt(1,:,[3]),'m:');
hold off;

%phs1=plot(subspaces, avgHSubspaceInt(1,:,[1]),'b:');
%phs2=plot(subspaces, avgHSubspaceStat(1,:,1),'b+:');
%phs3=plot(subspaces, avgHSubspaceInt(1,:,[3]),'b:');

% %pp1=plot(subspaces, pcaInt(:,[1 3]),'m:');
% %pp2=plot(subspaces, pcaInt(:,2),'mx:');

title('95% bounds for various projections');
legend([pr2 ps2 ppc2 psr2],'Gaussian','Subspace','PCA','Achlioptas');
hold off;
 figure

     centers=0.49:0.01:1.51;
     normPCAT=reshape(permute(normPCA,[2 1 3]),1,size(subspaces,2),[]);
     normSubT=reshape(permute(normPX,[2 1 3]),1,size(subspaces,2),[]);
     normRPT=reshape(permute(normRX,[2 1 3]),1,size(subspaces,2),[]);
     normSRPT=reshape(permute(normSRX,[2 1 3]),1,size(subspaces,2),[]);
     normHSubT=reshape(permute(normHPX,[2 1 3]),1,size(subspaces,2),[]);
     
      [PXNHist, binCenter1]=hist(reshape(normSubT(1,20,:),1,[]),centers);
      [RXNHist, binCenter2]=hist(reshape(normRPT(1,20,:),1,[]),centers);
      [PCANHist, binCenter3]=hist(reshape(normPCAT(1,20,:),1,[]),centers);
% %      [PCANHist, binCenter4]=hist(normPCA,centers);
      [SRXNHist, binCenter5]=hist(reshape(normSRPT(1,20,:),1,[]),centers);
      [HPXNHist, binCenter6]=hist(reshape(normHSubT(1,20,:),1,[]),centers);
      
         plot(binCenter1,PXNHist, 'b-');
         hold on;
         title(['Number of projections ',num2str(subspaces(20))]);
         plot(binCenter2,RXNHist, 'r-');
         plot(binCenter3,PCANHist, 'g-');
         plot(binCenter5,SRXNHist, 'm-');
         plot(binCenter6,HPXNHist, 'c-');
% %       plot(binCenter4,PCANHist, 'm-');
         grid on;
         legend('subspace','gaussian','PCA','Achlioptas', 'Householder');
         hold off;
         figure


   avgRPRunTime=mean(rpRunTime,1);
avgSRPRunTime=mean(srpRunTime,1);
avgSubspaceRunTime=mean(subspaceRunTime,1);
avgHSubspaceRunTime=mean(hsubspaceRunTime,1);

avgPCARunTime=mean(pcaRunTime,1)+mean(PCATime);
figure

pr1=plot(subspaces, avgRPRunTime,'k-s');
hold on
title(['Runtime vs Projections ']);
ps1=plot(subspaces, avgSubspaceRunTime,'k-o');
ppc1=plot(subspaces, avgPCARunTime,'k-+');
psr1=plot(subspaces, avgSRPRunTime,'k-*');
%phs1=plot(subspaces, avgHSubspaceRunTime,'k-');
         grid on;
legend([pr1 ps1 ppc1 psr1],'Gaussian','Subspace','PCA','Achlioptas');
         hold off;
         
               
         
         

         
 subspaceIdx=10;
 binCount=100;
 tempSubspace=reshape(normSubT(1,subspaceIdx,:),1,[]);
 tempRP=reshape(normRPT(1,subspaceIdx,:),1,[]);
  tempPCA=reshape(normPCAT(1,subspaceIdx,:),1,[]);
% %      [PCANHist, binCenter4]=hist(normPCA,centers);
 tempSRP=reshape(normSRPT(1,subspaceIdx,:),1,[]);
tempHSubspace=reshape(normHSubT(1,subspaceIdx,:),1,[]);
figure;   
ax=axes;

 hist(ax,tempSubspace,binCount);
      title(['Number of projections ',num2str(subspaces(subspaceIdx))]);
      hold on;
      hist(ax,tempRP,binCount);
      hist(ax,tempPCA,binCount);
   % %      [PCANHist, binCenter4]=hist(normPCA,centers);
      hist(ax,tempSRP,binCount);
      hist(ax,tempHSubspace,binCount);
      h = findobj(ax,'Type','patch');
            set(h(1),'FaceColor','c','EdgeColor','c','facealpha',0.5,'edgealpha',0.4);
            set(h(2),'FaceColor','m','EdgeColor','m','facealpha',0.5,'edgealpha',0.4);
            set(h(3),'FaceColor','g','EdgeColor','g','facealpha',0.5,'edgealpha',0.4);
            set(h(4),'FaceColor','r','EdgeColor','r','facealpha',0.5,'edgealpha',0.4);
            set(h(5),'FaceColor','b','EdgeColor','b','facealpha',0.5,'edgealpha',0.4);
            grid on;
            legend('subspace','gaussian','PCA','Achlioptas', 'Householder');  
      hold off;
            figure;
      
      
      [kdensSub,xSub]=ksdensity(tempSubspace);
      [kdensRP,xRP]=ksdensity(tempRP);
      [kdensPCA,xPCA]=ksdensity(tempPCA);
      [kdensSRP,xSRP]=ksdensity(tempSRP);
      [kdensHSub,xHSub]=ksdensity(tempHSubspace);
      subplot(2,2,2)
       [tempMax,tempX] =  hist(tempSubspace,binCount);
 tempScale=max(tempMax)/max(kdensSub);
      
       plot(xSub,kdensSub*tempScale, 'k-');
       hold on
        hist(tempSubspace,binCount);
       title(['Kernel Density Plot for ',num2str(subspaces(subspaceIdx)),' projections.']);

       %hold on;
 
   % %      [PCANHist, binCenter4]=hist(normPCA,centers);
  
%      hist(ax,tempHSubspace,binCount);

       subplot(2,2,1)
        [tempMax,tempX] =  hist(tempRP,binCount);
 tempScale=max(tempMax)/max(kdensRP);
       plot(xRP,kdensRP*tempScale, 'k-');
       hold on
                 hist(tempRP,binCount);
   
       subplot(2,2,3)
          [tempMax,tempX] =  hist(tempPCA,binCount);
 tempScale=max(tempMax)/max(kdensPCA);
       plot(xPCA,kdensPCA*tempScale, 'k-');
       hold on
            hist(tempPCA,binCount);
       subplot(2,2,4)
       plot(xSRP,kdensSRP, 'k-');
       [tempMax,tempX] =  hist(tempSRP,binCount);
 tempScale=max(tempMax)/max(kdensSRP);
       hold on
           hist(tempSRP,binCount);
 %      plot(xHSub,kdensHSub, 'c-');
      
       
       

% %       plot(binCenter4,PCANHist, 'm-');
      grid on;
         legend('subspace','gaussian','PCA','Achlioptas');
         hold off;


      
      
      
      