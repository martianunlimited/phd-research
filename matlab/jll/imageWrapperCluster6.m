
%Define definition of the run, including range of subspaces, files to run,
%number of pairs of vectors, window size and number of random windows
files=dir('*.tiff');
fileCount=size(files,1);
runCount=500;
windSize=50;
clusters=[1 7];
clusterCount=size(clusters,2);
subspaces=[5 10 20 35 50 75 100 150 200 350 500 700];
subspaceCount=size(subspaces,2);
pairs=100;

tic;

%% Initialize all statistics to 0
subspaceInt=zeros(fileCount,clusterCount,subspaceCount,3);
%hsubspaceInt=zeros(fileCount,clusterCount,subspaceCount,3);
RPInt=zeros(fileCount,subspaceCount,3);
SRPInt=zeros(fileCount,subspaceCount,3);
%PCAInt=zeros(fileCount,clusterCount,subspaceCount,3);
%DCTInt=zeros(fileCount,subspaceCount,3);

subspaceStat=zeros(fileCount,clusterCount,subspaceCount,4);
%hsubspaceStat=zeros(fileCount,subspaceCount,4);
RPStat=zeros(fileCount,subspaceCount,4);
SRPStat=zeros(fileCount,subspaceCount,4);
%PCAStat=zeros(fileCount,clusterCount,subspaceCount,4);
%DCTStat=zeros(fileCount,subspaceCount,4);

subspaceRunTime=zeros(fileCount,clusterCount,subspaceCount);
%hsubspaceRunTime=zeros(fileCount,clusterCount,subspaceCount);
rpRunTime=zeros(fileCount,subspaceCount);
srpRunTime=zeros(fileCount,subspaceCount);
%pcaRunTime=zeros(fileCount,clusterCount,subspaceCount);
%PCATime=zeros(fileCount,1);
%HPreTime=zeros(fileCount,1);

normPX=zeros(fileCount,clusterCount, subspaceCount,pairs*(pairs-1)/2);
normRX=zeros(fileCount, subspaceCount, pairs*(pairs-1)/2);
normSRX=zeros(fileCount, subspaceCount, pairs*(pairs-1)/2);
%normPCA=zeros(fileCount,clusterCount, subspaceCount,pairs*(pairs-1)/2);
%normDCT=zeros(fileCount,subspaceCount,pairs*(pairs-1)/2);
%normHPX=zeros(fileCount,subspaceCount,pairs*(pairs-1)/2);

%% Start of the Outer loop 
for fileNo=1:1:fileCount
% Loop through each file and store data as a data matrix
filename=files(fileNo).name
X=imread(filename);
X=double(X);
sqrt(max(max(X.^2/mean(mean(X.^2)))))

% Randomly generate $runCount windows of size $windSize and store the
% vectors as Xvec

sizeY=size(X,2)
sizeX=size(X,1)
coorXY=zeros(runCount,2);
Xvec=zeros(runCount,windSize^2);
s=rng;
rng(12486);
for k=1:runCount
coorXY(k,:)=[randi([1 sizeX-windSize]) randi([1 sizeY-windSize])];
Xvec(k,:)=reshape(X(coorXY(k,1):coorXY(k,1)+windSize-1,coorXY(k,2):coorXY(k,2)+windSize-1),1,[]);
% Experiment 1: Normalize the vectors to have norm 1 
% Xvec(k,:)=Xvec(k,:)/norm(Xvec(k,:));
end
% Experiment 2: Mean centering;
% Xvec=Xvec-mean(mean(Xvec));
%
randInd=randperm(runCount,pairs);
rng(s)
for k=1:size(subspaces,2)
        subspace=subspaces(k)
        R=randproj(subspace,windSize^2);
        SR=sparse(randproj_3(subspace,windSize^2));
        index=0;        
        for i = 1:pairs
            for j= i+1 : pairs
                index=index+1;
                u=Xvec(randInd(i),:);
                v=Xvec(randInd(j),:);
                uv=(u-v)';
                dist=norm(uv);    
                if (dist==0) 
                    continue;      
                end  
                uvR=uv'*R';
                normRX(fileNo,k,index)=norm(uvR)/dist*sqrt(windSize^2/subspace);
                uvSR=uv'*SR';
                normSRX(fileNo,k,index)=norm(uvSR)/dist*sqrt(1/subspace);
            end
        end
        
        RPInt(fileNo,k,:)=prctile(normRX(fileNo,k,:),[5 50 95]);
        SRPInt(fileNo,k,:)=prctile(normSRX(fileNo,k,:),[5 50 95]);
        RPStat(fileNo,k,:)=[mean(normRX(fileNo,k,:)) var(normRX(fileNo,k,:)) min(normRX(fileNo,k,:)) max(normRX(fileNo,k,:))];
        SRPStat(fileNo,k,:)=[mean(normSRX(fileNo,k,:)) var(normSRX(fileNo,k,:)) min(normSRX(fileNo,k,:)) max(normSRX(fileNo,k,:))];
        
        
end
    meanXvec=mean(Xvec.^2);
    z=linkage(meanXvec','ward','euclidean');
    figure; dendrogram(z); title(filename);
for l=1:clusterCount
    clustNo=clusters(l)
    c=cell(1,clustNo);
    s=cell(1,clustNo);
%    varXvec=var(Xvec);
%    meanXvec=mean(Xvec.^2);
%    [cent clust count]=simpleKMeans2(varXvec',clustNo);
%    clust=kmeans(meanXvec',clustNo);
    clust=cluster(z,'maxclust',clustNo);
    totalWeight=0;
    for m=1:clustNo
        c{m}=find(clust==m);
        s{m}=sum(clust==m);
        totalWeight=totalWeight+s{m};
 %       s{m}=sqrt(var(clust(c{m},:)'));
 %       totalWeight=totalWeight+sum(s{m});
    end

    for k=1:size(subspaces,2)
        km=cell(1,clustNo);
        pm=cell(1,clustNo);
        subspace=subspaces(k);


        index=0;
        for i = 1:pairs
            for j= i+1 : pairs
                p=[];
                subspace_r=0;
                for m=1:clustNo
                    km{m}=round(sum(s{m}/totalWeight)*subspace);
                    pm{m}=datasample(c{m},km{m},'Replace',false);
                    p=[p pm{m}'];
                    subspace_r=subspace_r+km{m};
                end
                
                
                index=index+1;
                u=Xvec(randInd(i),:);
                v=Xvec(randInd(j),:);
                uv=(u-v)';
                dist=norm(uv);    
                if (dist==0) 
                    continue;      
                end  
                uvP=[uv(p)];
                normPX(fileNo,l,k,index)=norm(uvP)/dist*sqrt(windSize^2/subspace_r);
            end
        end
        subspaceStat(fileNo,l,k,:)=[mean(normPX(fileNo,l,k,:)) var(normPX(fileNo,l,k,:)) min(normPX(fileNo,l,k,:)) max(normPX(fileNo,l,k,:))];
        subspaceInt(fileNo,l,k,:)=prctile(normPX(fileNo,l,k,:),[5 50 95]);
    end
end

end

avgRPStat=mean(RPStat,1);
avgSRPStat=mean(SRPStat,1);
avgSubspaceStat=mean(subspaceStat,1);
avgHSubspaceStat=mean(hsubspaceStat,1);
%avgPCAStat=mean(PCAStat,1);

avgRPInt=mean(RPInt,1);
avgSRPInt=mean(SRPInt,1);
avgSubspaceInt=mean(subspaceInt,1);
avgHSubspaceInt=mean(hsubspaceInt,1);
%avgPCAInt=mean(PCAInt,1);

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
%ppc1=plot(subspaces, avgPCAInt(1,:,[1]),'g:');
hold on;
%ppc2=plot(subspaces, avgPCAStat(1,:,1),'go:');
%ppc3=plot(subspaces, avgPCAInt(1,:,[3]),'g:');
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
         legend('subspace','gaussian','PCA','Achlioptas', 'Householder');
         hold off;


      
      
      
      for fileNo=1:10
          for k=1:120
              subspaceStat(fileNo,k,:)=[mean(normPX(fileNo,k,:)) var(normPX(fileNo,k,:)) min(normPX(fileNo,k,:)) max(normPX(fileNo,k,:))];
              subspaceInt(fileNo,k,:)=prctile(normPX(fileNo,k,:),[5 50 95]);
          end
      end
      
      
      
      avgStat11=mean(subspaceStat11,1);
      avgStat7=mean(subspaceStat7,1);
      avgStat3=mean(subspaceStat3,1);
      avgStat11=reshape(avgStat11,120,[]);
      avgStat7=reshape(avgStat7,120,[]);
      avgStat3=reshape(avgStat3,120,[]);
      
      avgInt11=mean(subspaceInt11,1);
      avgInt7=mean(subspaceInt7,1);
      avgInt3=mean(subspaceInt3,1);
      avgInt11=reshape(avgInt11,120,[]);
      avgInt7=reshape(avgInt7,120,[]);
      avgInt3=reshape(avgInt3,120,[]);      
      
      
      
      
      centers=0.5:0.02:1.5;
      
      for i=1:23
          figure;
          subplot(4,6,1);
          hist(reshape(normRX(i,3,:),1,[]),centers);
          title(files(i).name);
          xlim([0.75 1.25]);
          for j=1:20
              subplot(4,6,j+1);
              hist(reshape(normPX(i,j,3,:),1,[]),centers);
              title(sprintf('noClust= %d', clusters(j)));
              xlim([0.75 1.25]);
          end
      end
      
      stats=zeros(23,clusterCount+1,subspaceCount,9);
      k=9;
      for k=1:subspaceCount
      for i=1:23
              stats(i,1,k,1)=prctile(normRX(i,k,:),95)-prctile(normRX(i,k,:),5);
              stats(i,1,k,2)=prctile(normRX(i,k,:),84)-prctile(normRX(i,k,:),16);
              stats(i,1,k,3)=prctile(normRX(i,k,:),97.5)-prctile(normRX(i,k,:),2.5);
              stats(i,1,k,4)=prctile(normRX(i,k,:),99.85)-prctile(normRX(i,k,:),0.15);
              stats(i,1,k,5)=prctile(normRX(i,k,:),100)-prctile(normRX(i,k,:),0);
              stats(i,1,k,6)=sqrt(var(normRX(i,k,:)));
              stats(i,1,k,7)=prctile(normRX(i,k,:),50);
              stats(i,1,k,8)=mean(normRX(i,k,:));
              stats(i,1,k,9)=sqrt(var(normPX(i,j,k,:)));
              
          for j=1:clusterCount
              stats(i,j+1,k,1)=prctile(normPX(i,j,k,:),95)-prctile(normPX(i,j,k,:),5);
              stats(i,j+1,k,2)=prctile(normPX(i,j,k,:),84)-prctile(normPX(i,j,k,:),16);
              stats(i,j+1,k,3)=prctile(normPX(i,j,k,:),97.5)-prctile(normPX(i,j,k,:),2.5);
              stats(i,j+1,k,4)=prctile(normPX(i,j,k,:),99.85)-prctile(normPX(i,j,k,:),0.15);
              stats(i,j+1,k,5)=prctile(normPX(i,j,k,:),100)-prctile(normPX(i,j,k,:),0);
              stats(i,j+1,k,6)=sqrt(var(normPX(i,j,k,:)));
              stats(i,j+1,k,7)=prctile(normPX(i,j,k,:),50);
              stats(i,j+1,k,8)=mean(normPX(i,j,k,:));
              stats(i,j+1,k,9)=sqrt(var(normPX(i,j,k,:)))/sqrt(var(normPX(i,1,k,:)));
          end
      end
      end
      
figure;
for fileNo=1:fileCount
    filename=files(fileNo).name;
    subplot(4,6,fileNo);
    plot([1:40],stats(fileNo,[2:41],3,6));
    title(filename);
end
      


      
for fileNo=1:1:fileCount
% Loop through each file and store data as a data matrix
filename=files(fileNo).name
X=imread(filename);
X=double(X);
sqrt(max(max(X.^2/mean(mean(X.^2)))))

% Randomly generate $runCount windows of size $windSize and store the
% vectors as Xvec

sizeY=size(X,2)
sizeX=size(X,1)
coorXY=zeros(runCount,2);
Xvec=zeros(runCount,windSize^2);
s=rng;
rng(12486);
for k=1:runCount
coorXY(k,:)=[randi([1 sizeX-windSize]) randi([1 sizeY-windSize])];
Xvec(k,:)=reshape(X(coorXY(k,1):coorXY(k,1)+windSize-1,coorXY(k,2):coorXY(k,2)+windSize-1),1,[]);
% Experiment 1: Normalize the vectors to have norm 1 
% Xvec(k,:)=Xvec(k,:)/norm(Xvec(k,:));
end
%meanXvec=Xvec.^2./mean(Xvec.^2);
meanXvec=mean(Xvec.^2);
z=linkage(meanXvec','ward','euclidean');
figure
dendrogram(z);
end  


figure;
pr1=plot(subspaces, avgRPInt(1,:,[1]),'r:s');
hold on
pr2=plot(subspaces, avgRPStat(1,:,1),'k:s');
pr3=plot(subspaces, avgRPInt(1,:,[3]),'r:s');

ps1=plot(subspaces, a(1,:,[1]),'b:+');
ps2=plot(subspaces, a(1,:,[2]),'k+:');
ps3=plot(subspaces, a(1,:,[3]),'b:+');

ps71=plot(subspaces, a(2,:,[1]),'g:o');
ps72=plot(subspaces, a(2,:,[2]),'k+:');
ps73=plot(subspaces, a(2,:,[3]),'g:o');

legend([pr1 ps1 ps71],'Gaussian','Subspace','Stratified Subspace');
title('95% bounds for various projections');
xlim([0 600]);
ylim([0 2]);
ylabel('$\frac{\|P(X_j-X_i)\|}{\|X_j-X_i\|}$','Interpreter','Latex')
xlabel('Projection Dimension(k)');
hold off;
