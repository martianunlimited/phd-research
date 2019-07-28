prob=[0.02 0.035 0.05 0.10 0.15 0.25 0.5];
dims=[500 1000 2500 5000 10000 25000];
probCount=size(prob,2);
dimCount=size(dims,2);
subspaces=[5 7 10 15 25 35 50 70 100 150 250 350 500];
subspaceCount=size(subspaces,2);
pairs=400;
obs=400;


subspaceStat=zeros(probCount,dimCount,subspaceCount,4);
subspace2Stat=zeros(probCount,dimCount,subspaceCount,4);
subspace3Stat=zeros(probCount,dimCount,subspaceCount,4);

hsubspaceStat=zeros(probCount,dimCount,subspaceCount,4);
rpStat=zeros(probCount,dimCount,subspaceCount,4);
srpStat=zeros(probCount,dimCount,subspaceCount,4);
pcaStat=zeros(probCount,dimCount,subspaceCount,4);


subspaceInt=zeros(probCount,dimCount,subspaceCount,3);
subspace2Int=zeros(probCount,dimCount,subspaceCount,3);
subspace3Int=zeros(probCount,dimCount,subspaceCount,3);

hsubspaceInt=zeros(probCount,dimCount,subspaceCount,3);
rpInt=zeros(probCount,dimCount,subspaceCount,3);
srpInt=zeros(probCount,dimCount,subspaceCount,3);

subspaceRunTime=zeros(probCount,dimCount,subspaceCount);
subspaceRunTime2=zeros(probCount,dimCount,subspaceCount);
subspaceRunTime3=zeros(probCount,dimCount,subspaceCount);

hsubspaceRunTime=zeros(probCount,dimCount,subspaceCount);
hsubspaceRunTime2=zeros(probCount,dimCount,subspaceCount);
rpRunTime=zeros(probCount,dimCount,subspaceCount);
srpRunTime=zeros(probCount,dimCount,subspaceCount);
pcaRunTime=zeros(probCount,dimCount,subspaceCount);

pcaInt=zeros(probCount,dimCount,subspaceCount,3);

%scaleFactor=zeros(fileCount,1);
fileNo=0;

normSub=zeros(probCount,dimCount,subspaceCount,pairs*(pairs-1)/2);
normHSub=zeros(probCount,dimCount,subspaceCount,pairs*(pairs-1)/2);
normRP=zeros(probCount,dimCount,subspaceCount,pairs*(pairs-1)/2);
normSRP=zeros(probCount,dimCount,subspaceCount,pairs*(pairs-1)/2);

for probNo=1:probCount
    for dimNo = 1:dimCount
        baseProb=prob(probNo)
        dim=dims(dimNo)
        

  
        
S=binRandGen(baseProb,dim,obs);

%S=S(sum(S,2)>0,:);
dim=size(S,1);
obs=size(S,2);
tic
mn=mean(S,2);
v=ones(1,dim)*sqrt(1/dim);
v=v/norm(v);
%[U,Sig,eigV]=svds((S-repmat(mn,1,obs))'/sqrt(obs-1),obs);
toc
alpha=full(v*S);
%scale=sum(var(S'.^2))*dim/((sum(sum(S.^2))/obs)^2)/4
%scaleFactor(fileNo)=scale;
%scale2=sum(var(S'.^2))*dim/((sum(sum(S.^2))/obs)^2);
scale2=1;
scale=1;
%v=v*sqrt(scale);
%scale=ceil(sum(var(S'.^2))/sum(var(S.^2))*1/6*(3-sqrt(3)))
%scale=22
for k=1:size(subspaces,2)    

    subspace=subspaces(k)
    randIdx=randperm(obs,pairs);

l=0;
tic;
SR=randproj_3(subspace,dim);
 SS= ((SR>0) - (SR<0)) ;
 SRindex=cell(subspace,2);
 for subI=1:subspace;
     SRindex{subI,1}=find(SS(subI,:)==-1);
     SRindex{subI,2}=find(SS(subI,:)==1);
 end
%SX=zeros(subspace,obs);
%for subI=1:subspace
%   SX(subI,:)=sqrt(3)*(sum(S(SRindex{subI,2},:))-sum(S(SRindex{subI,1},:)));
%end
SP=SR*S;
for i=1:pairs
    for j=i+1:pairs
        l=l+1;
        uv=S(:,randIdx(i))-S(:,randIdx(j));
        uvP=SP(:,randIdx(i))-SP(:,randIdx(j));
%        Suv=SX(:,randIdx(i))-SX(:,randIdx(j));
%        uvP=Suv;
        normSRP(probNo,dimNo,k,l)=sqrt(sum((uvP).^2)/ sum(uv.^2)/subspace);
%       normPCA(l)=sqrt(norm(eigV(:,1:subspace)'*uv)^2/ sum(uv.^2));
       
    end
end
srpRunTime(probNo,dimNo,k)=toc;


l=0;
tic;
P=randperm(dim,ceil(subspace*scale2));
P=sort(P);
SP=S(P,:);
for i=1:pairs
    for j=i+1:pairs
        l=l+1;
        uv=S(:,randIdx(i))-S(:,randIdx(j));
        uvP=SP(:,randIdx(i))-SP(:,randIdx(j));   
        normSub(probNo,dimNo,k,l)=sqrt(sum(uvP.^2) / sum(uv.^2)*dim/ceil(subspace*scale2));
    end
end
subspaceRunTime(probNo,dimNo,k)=toc;
 
% l=0;
% tic;
% P=randperm(dim,ceil(subspace*scale2));
% P=sort(P);
% for i=1:pairs
%     for j=i+1:pairs
%         
%         l=l+1;
%         uv=full(S(:,randIdx(i))-S(:,randIdx(j)));          
%         normSub2(fileNo,k,l)=sqrt(sum(uv(P).^2) / sum(uv.^2)*dim/ceil(subspace*scale2));
% 
%     end
% end
% subspaceRunTime2(fileNo,k)=toc;
 
 %l=0;
 %tic;
 %P=randperm(dim,ceil(subspace));
 %P=sort(P);
 %for i=1:pairs
 %    for j=i+1:pairs
 %        
 %        l=l+1;
 %       uv=full(S(:,randIdx(i))-S(:,randIdx(j)));          
 %        normSub3(fileNo,k,l)=sqrt(sum(uv(P).^2) / sum(uv.^2)*dim/ceil(subspace));
% 
%     end
% end
% subspaceRunTime3(fileNo,k)=toc;


l=0;
tic;

for i=1:pairs
    for j=i+1:pairs
        l=l+1;
        uv=full(S(:,randIdx(i))-S(:,randIdx(j)));
        uvP=abs(uv(P));
        %alpha2=alpha(randIdx(i))-alpha(randIdx(j));
        uvS=sum(abs(uv));
        uvP=uvP-2*uvS/dim;
        %beta=v(P)*uv(P);
        normHSub(probNo,dimNo,k,l)=sqrt((sum(uvP.^2)) / sum(uv.^2)*dim/ceil(subspace*scale2));
    end
end
hsubspaceRunTime(probNo,dimNo,k)=toc;

% l=0;
% tic;
% P=randperm(dim,ceil(subspace*scale));
% P=sort(P);
% for i=1:pairs
%     for j=i+1:pairs
%         l=l+1;
%         uv=full(S(:,randIdx(i))-S(:,randIdx(j)));  
%         alpha2=alpha(randIdx(i))-alpha(randIdx(j));
%         %beta=v(P)*uv(P);
%         uv2=uv-2*alpha2*v';
%         normHSub2(fileNo,k,l)=sqrt((sum(uv2(P).^2)) / sum(uv.^2)*dim/subspace/scale);
%     end
% end
% hsubspaceRunTime2(fileNo,k)=toc;


l=0;
tic;
R=randproj(subspace,dim);
SP=R*S;
for i=1:pairs
    for j=i+1:pairs
        l=l+1;
        uv=S(:,randIdx(i))-S(:,randIdx(j)); 
        uvP=SP(:,randIdx(i))-SP(:,randIdx(j));
       normRP(probNo,dimNo,k,l)=sqrt(sum(uvP.^2)/ sum(uv.^2)/subspace);
    end
end
rpRunTime(probNo,dimNo,k)=toc;
% l=0;
% %tic;
% for i=1:pairs
%     for j=i+1:pairs
%         l=l+1;
% %        uv=S(:,randIdx(i))-S(:,randIdx(j));    
% %        normPCA(l)=sqrt(norm(eigV(:,1:subspace)'*uv)^2/ sum(uv.^2));
%        
%     end
% end
% %pcaRunTime(k)=toc;



subspaceStat(probNo,dimNo,k,:)=[mean(normSub(probNo,dimNo,k,:)) var(normSub(probNo,dimNo,k,:)) min(normSub(probNo,dimNo,k,:)) max(normSub(probNo,dimNo,k,:))];

hsubspaceStat(probNo,dimNo,k,:)=[mean(normHSub(probNo,dimNo,k,:)) var(normHSub(probNo,dimNo,k,:)) min(normHSub(probNo,dimNo,k,:)) max(normHSub(probNo,dimNo,k,:))];
rpStat(probNo,dimNo,k,:)=[mean(normRP(probNo,dimNo,k,:)) var(normRP(probNo,dimNo,k,:)) min(normRP(probNo,dimNo,k,:)) max(normRP(probNo,dimNo,k,:))];
srpStat(probNo,dimNo,k,:)=[mean(normSRP(probNo,dimNo,k,:)) var(normSRP(probNo,dimNo,k,:)) min(normSRP(probNo,dimNo,k,:)) max(normSRP(probNo,dimNo,k,:))];

%pcaStat(k,:)=[mean(normPCA(:)) var(normPCA(:)) min(normPCA(:)) max(normPCA(:))];
subspaceInt(probNo,dimNo,k,:)=prctile(normSub(probNo,dimNo,k,:),[5 50 95]);
hsubspaceInt(probNo,dimNo,k,:)=prctile(normHSub(probNo,dimNo,k,:),[5 50 95]);
rpInt(probNo,dimNo,k,:)=prctile(normRP(probNo,dimNo,k,:),[5 50 95]);
srpInt(probNo,dimNo,k,:)=prctile(normSRP(probNo,dimNo,k,:),[5 50 95]);


% %pcaInt(k,:)=prctile(normPCA,[5 50 95]);

end

    end

 
      centers=0.0:0.01:2;
 
       [PXNHist, binCenter1]=hist(reshape(normSub(probNo,:,7,:),1,[]),centers);
       [RXNHist, binCenter2]=hist(reshape(normRP(probNo,:,7,:),1,[]),centers);
       [HPXNHist, binCenter3]=hist(reshape(normHSub(probNo,:,7,:),1,[]),centers);
   %    [PCANHist, binCenter4]=hist(normPCA,centers);
              [SRXNHist, binCenter5]=hist(reshape(normSRP(probNo,:,7,:),1,[]),centers);
          plot(binCenter1,PXNHist, 'b-');
          hold on;
          title(['Sparse: Number of projections Prob: ' num2str(baseProb) ]);
          plot(binCenter2,RXNHist, 'r-');
          plot(binCenter3,HPXNHist, 'g-');
          plot(binCenter5,SRXNHist, 'm-');
%          plot(binCenter4,PCANHist, 'm-');
          grid on;
          legend('subspace','gaussian','householder','Archiolptas');
          xlim([0.5 1.5])
          hold off;
          figure

    
pr1=plot(subspaces, reshape(rpInt(probNo,1,:,1),1,[]),'r:');
hold on
pr2=plot(subspaces, reshape(rpStat(probNo,1,:,1),1,[]),'r.:');
pr3=plot(subspaces, reshape(rpInt(probNo,1,:,3),1,[]),'r:');

ps1=plot(subspaces, reshape(subspaceInt(probNo,1,:,1),1,[]),'b:');
ps2=plot(subspaces, reshape(subspaceStat(probNo,1,:,1),1,[]),'b+:');
ps3=plot(subspaces, reshape(subspaceInt(probNo,1,:,3),1,[]),'b:');

phs1=plot(subspaces, reshape(hsubspaceInt(probNo,1,:,1),1,[]),'g:');
phs2=plot(subspaces, reshape(hsubspaceStat(probNo,1,:,1),1,[]),'go:');
phs3=plot(subspaces, reshape(hsubspaceInt(probNo,1,:,3),1,[]),'g:');

psr1=plot(subspaces, reshape(srpInt(probNo,1,:,1),1,[]),'m:');
psr2=plot(subspaces, reshape(srpStat(probNo,1,:,1),1,[]),'mx:');
psr3=plot(subspaces, reshape(srpInt(probNo,1,:,3),1,[]),'m:');

%pp1=plot(subspaces, pcaInt(:,[1 3]),'m:');
%pp2=plot(subspaces, pcaInt(:,2),'mx:');

title(['Sparse: 95% bounds for various projections smallest dim (2500) p=' num2str(prob(probNo))]);
legend([pr2 ps2 phs2 psr2],'Gaussian','Subspace','Householder','Archiolptas');
ylim([0 2]);
hold off;
figure  
    
pr1=plot(subspaces, reshape(rpInt(probNo,dimCount,:,1),1,[]),'r:');
hold on
pr2=plot(subspaces, reshape(rpStat(probNo,dimCount,:,1),1,[]),'r.:');
pr3=plot(subspaces, reshape(rpInt(probNo,dimCount,:,3),1,[]),'r:');

ps1=plot(subspaces, reshape(subspaceInt(probNo,dimCount,:,1),1,[]),'b:');
ps2=plot(subspaces, reshape(subspaceStat(probNo,dimCount,:,1),1,[]),'b+:');
ps3=plot(subspaces, reshape(subspaceInt(probNo,dimCount,:,3),1,[]),'b:');

phs1=plot(subspaces, reshape(hsubspaceInt(probNo,dimCount,:,1),1,[]),'g:');
phs2=plot(subspaces, reshape(hsubspaceStat(probNo,dimCount,:,1),1,[]),'go:');
phs3=plot(subspaces, reshape(hsubspaceInt(probNo,dimCount,:,3),1,[]),'g:');

psr1=plot(subspaces, reshape(srpInt(probNo,dimCount,:,1),1,[]),'m:');
psr2=plot(subspaces, reshape(srpStat(probNo,dimCount,:,1),1,[]),'mx:');
psr3=plot(subspaces, reshape(srpInt(probNo,dimCount,:,3),1,[]),'m:');

%pp1=plot(subspaces, pcaInt(:,[1 3]),'m:');
%pp2=plot(subspaces, pcaInt(:,2),'mx:');

title(['Sparse: 95% bounds for various projections largest dim (10000) p=' num2str(prob(probNo))]);
legend([pr2 ps2 phs2 psr2],'Gaussian','Subspace','Householder','Archiolptas');
ylim([0 2]);
hold off;
figure  
     
    
    
    
pr1=plot(subspaces, reshape(mean(rpInt(probNo,:,:,1),2),1,[]),'r:');
hold on
pr2=plot(subspaces, reshape(mean(rpStat(probNo,:,:,1),2),1,[]),'r.:');
pr3=plot(subspaces, reshape(mean(rpInt(probNo,:,:,3),2),1,[]),'r:');

ps1=plot(subspaces, reshape(mean(subspaceInt(probNo,:,:,1),2),1,[]),'b:');
ps2=plot(subspaces, reshape(mean(subspaceStat(probNo,:,:,1),2),1,[]),'b+:');
ps3=plot(subspaces, reshape(mean(subspaceInt(probNo,:,:,3),2),1,[]),'b:');

phs1=plot(subspaces, reshape(mean(hsubspaceInt(probNo,:,:,1),2),1,[]),'g:');
phs2=plot(subspaces, reshape(mean(hsubspaceStat(probNo,:,:,1),2),1,[]),'go:');
phs3=plot(subspaces, reshape(mean(hsubspaceInt(probNo,:,:,3),2),1,[]),'g:');

psr1=plot(subspaces, reshape(mean(srpInt(probNo,:,:,1),2),1,[]),'m:');
psr2=plot(subspaces, reshape(mean(srpStat(probNo,:,:,1),2),1,[]),'mx:');
psr3=plot(subspaces, reshape(mean(srpInt(probNo,:,:,3),2),1,[]),'m:');

%pp1=plot(subspaces, pcaInt(:,[1 3]),'m:');
%pp2=plot(subspaces, pcaInt(:,2),'mx:');

title(['Sparse: 95% bounds for various projections avged all dim p=' num2str(prob(probNo))]);
legend([pr2 ps2 phs2 psr2],'Gaussian','Subspace','Householder','Archiolptas');
ylim([0 2]);
hold off;
figure
end
% 
% avgRPStat=mean(rpStat,1);
% avgSRPStat=mean(srpStat,1);
% avgSubspaceStat=mean(subspaceStat,1);
% avgHSubspaceStat=mean(hsubspaceStat,1);
% 
% avgRPInt=mean(rpInt,1);
% avgSRPInt=mean(srpInt,1);
% avgSubspaceInt=mean(subspaceInt,1);
% avgHSubspaceInt=mean(hsubspaceInt,1);
% figure
% pr1=plot(subspaces, avgRPInt(1,:,[1]),'r:');
% hold on
% pr2=plot(subspaces, avgRPStat(1,:,1),'r.:');
% pr3=plot(subspaces, avgRPInt(1,:,[3]),'r:');
% 
% ps1=plot(subspaces, avgSubspaceInt(1,:,[1]),'b:');
% ps2=plot(subspaces, avgSubspaceStat(1,:,1),'b+:');
% ps3=plot(subspaces, avgSubspaceInt(1,:,[3]),'b:');
%  
% phs1=plot(subspaces, avgHSubspaceInt(1,:,[1]),'g:o');
% phs2=plot(subspaces, avgHSubspaceStat(1,:,1),'ko:');
% phs3=plot(subspaces, avgHSubspaceInt(1,:,[3]),'g:o');
% 
%  
% psr1=plot(subspaces, avgSRPInt(1,:,[1]),'m:');
% psr2=plot(subspaces, avgSRPStat(1,:,1),'mx:');
% psr3=plot(subspaces, avgSRPInt(1,:,[3]),'m:');
% hold off
% 
% % %pp1=plot(subspaces, pcaInt(:,[1 3]),'m:');
% % %pp2=plot(subspaces, pcaInt(:,2),'mx:');
% 
% figure;
% subplot(2,2,1);
% pr1=plot(subspaces, avgRPInt(1,:,[1]),'r:s');
% hold on
% pr2=plot(subspaces, avgRPStat(1,:,1),'k:s');
% pr3=plot(subspaces, avgRPInt(1,:,[3]),'r:s');
% hold off
% 
% subplot(2,2,2);
% ps1=plot(subspaces, avgSubspaceInt(1,:,[1]),'b:+');
% hold on;
% ps2=plot(subspaces, avgSubspaceStat(1,:,1),'k+:');
% ps3=plot(subspaces, avgSubspaceInt(1,:,[3]),'b:+');
% hold off;
% 
% subplot(2,2,3);
% phs1=plot(subspaces, avgHSubspaceInt(1,:,[1]),'g:o');
% hold on;
% phs2=plot(subspaces, avgHSubspaceStat(1,:,1),'ko:');
% phs3=plot(subspaces, avgHSubspaceInt(1,:,[3]),'g:o');
% hold off;
% 
% subplot(2,2,4);
% psr1=plot(subspaces, avgSRPInt(1,:,[1]),'m:x');
% hold on;
% psr2=plot(subspaces, avgSRPStat(1,:,1),'kx:');
% psr3=plot(subspaces, avgSRPInt(1,:,[3]),'mx:');
% hold off;
% 
% 
% 
% title('Sparse: 95% bounds for various projections');
% legend([pr2 ps2 phs2 psr2],'Gaussian','Subspace','Householder','Achlioptas');
% hold off;
%  figure
% 
%      centers=0.49:0.01:1.51;
%      normHSubT=reshape(permute(normHSub,[2 1 3]),1,size(subspaces,2),[]);
%      normSubT=reshape(permute(normSub,[2 1 3]),1,size(subspaces,2),[]);
%      normSub2T=reshape(permute(normSub2,[2 1 3]),1,size(subspaces,2),[]);
%      normSub3T=reshape(permute(normSub3,[2 1 3]),1,size(subspaces,2),[]);
% 
%      normRPT=reshape(permute(normRP,[2 1 3]),1,size(subspaces,2),[]);
%      normSRPT=reshape(permute(normSRP,[2 1 3]),1,size(subspaces,2),[]);
%      
%       [PXNHist, binCenter1]=hist(reshape(normSubT(1,10,:),1,[]),centers);
%       [RXNHist, binCenter2]=hist(reshape(normRPT(1,10,:),1,[]),centers);
%       [HPXNHist, binCenter3]=hist(reshape(normHSubT(1,10,:),1,[]),centers);
% % %      [PCANHist, binCenter4]=hist(normPCA,centers);
%       [SRXNHist, binCenter5]=hist(reshape(normSRPT(1,10,:),1,[]),centers);
%          plot(binCenter1,PXNHist, 'b-');
%          hold on;
%          title(['Sparse: Number of projections ',num2str(subspaces(10))]);
%          plot(binCenter2,RXNHist, 'r-');
%          plot(binCenter3,HPXNHist, 'g-');
%          plot(binCenter5,SRXNHist, 'm-');
% % %       plot(binCenter4,PCANHist, 'm-');
%          grid on;
%          legend('subspace','gaussian','householder','Achlioptas');
%          hold off;
%          figure
% % endavgRPRunTime=mean(rpRunTime,1)
% avgRPRunTime=mean(rpRunTime,1)
% avgSRPRunTime=mean(srpRunTime,1)
% avgSubspaceRunTime=mean(subspaceRunTime,1)
% avgHSubspace2RunTime=mean(hsubspaceRunTime2,1)
% avgHSubspaceRunTime=mean(hsubspaceRunTime,1)
% 
% figure
% 
% pr1=plot(subspaces, avgRPRunTime,'k-+');
% hold on
% title(['Runtime vs Projections ']);
% ps1=plot(subspaces, avgSubspaceRunTime,'k-o');
% %phs1=plot(subspaces, avgHSubspace2RunTime,'k-');
% phss1=plot(subspaces, avgHSubspaceRunTime,'k-h');
% psr1=plot(subspaces, avgSRPRunTime,'k-*');
%          grid on;
% legend([pr1 ps1 psr1 phss1],'Gaussian','Subspace','Achlioptas', 'Householder');
%          hold off;
% 
%          
%          
%     
%          
%          
%          
%          
%   subspaceIdx=5;
%  binCount=100;
%  tempSubspace=reshape(normSubT(1,subspaceIdx,:),1,[]);
%  tempSubspace2=reshape(normSub2T(1,subspaceIdx,:),1,[]);
%  tempSubspace3=reshape(normSub3T(1,subspaceIdx,:),1,[]);
% 
%  tempRP=reshape(normRPT(1,subspaceIdx,:),1,[]);
% %  tempPCA=reshape(normPCAT(1,subspaceIdx,:),1,[]);
% % %      [PCANHist, binCenter4]=hist(normPCA,centers);
%  tempSRP=reshape(normSRPT(1,subspaceIdx,:),1,[]);
% tempHSubspace=reshape(normHSubT(1,subspaceIdx,:),1,[]);
% figure;   
% %ax=axes;
% 
%       [kdensSub,xSub]=ksdensity(tempSubspace);
%       [kdensRP,xRP]=ksdensity(tempRP);
%       [kdensSub2,xSub2]=ksdensity(tempSubspace2);
%       [kdensSub3,xSub3]=ksdensity(tempSubspace3);
%       [kdensSRP,xSRP]=ksdensity(tempSRP);
%       [kdensHSub,xHSub]=ksdensity(tempHSubspace);
% 
% 
% subplot(2,2,2)
%  hist(tempSubspace2,binCount);
%  hold on;
%  [tempMax,tempX] =  hist(tempSubspace,binCount);
%  tempScale=max(tempMax)/max(kdensSub2);
%  plot(xSub2,kdensSub2*tempScale, 'k-');
%        title(['Number of projections ',num2str(subspaces(subspaceIdx))]);
%    %   hold on;
% hold off;   
% subplot(2,2,1)
% hist(tempRP,binCount);
% hold on
%  [tempMax,tempX] =  hist(tempRP,binCount);
%  tempScale=max(tempMax)/max(kdensRP);
%  plot(xRP,kdensRP*tempScale, 'k-');
%        title(['Number of projections ',num2str(subspaces(subspaceIdx))]);
%    %   hold on;
% hold off;   
% 
% 
% %      hist(tempSubspace3,binCount);
% 
% %      hist(tempSubspace2,binCount);
%    % %      [PCANHist, binCenter4]=hist(normPCA,centers);
% subplot(2,2,4)   
%       hist(tempSRP,binCount);
%       hold on
%       [tempMax,tempX] =  hist(tempSRP,binCount);
%  tempScale=max(tempMax)/max(kdensSRP);
%  plot(xSRP,kdensSRP*tempScale, 'k-');
%        title(['Number of projections ',num2str(subspaces(subspaceIdx))]);
%    %   hold on;
% hold off; 
%       
% subplot(2,2,3)      
%       hist(tempHSubspace,binCount);
%       
%        hold on;
%  [tempMax,tempX] =  hist(tempHSubspace,binCount);
%  tempScale=max(tempMax)/max(kdensHSub);
%  plot(xHSub,kdensHSub*tempScale, 'k-');
%        title(['Number of projections ',num2str(subspaces(subspaceIdx))]);
%    %   hold on;
% hold off;   
%       
%  %     h = findobj(ax,'Type','patch');
%  %           set(h(1),'FaceColor','c','EdgeColor','c','facealpha',0.5,'edgealpha',0.4);
%  %           set(h(2),'FaceColor','m','EdgeColor','m','facealpha',0.5,'edgealpha',0.4);
%  %           set(h(3),'FaceColor','g','EdgeColor','g','facealpha',0.5,'edgealpha',0.4);
%   %          set(h(4),'FaceColor','k','EdgeColor','k','facealpha',0.7,'edgealpha',0.7);
% 
%   %          set(h(5),'FaceColor','r','EdgeColor','r','facealpha',0.5,'edgealpha',0.4);
%   %          set(h(6),'FaceColor','b','EdgeColor','b','facealpha',0.5,'edgealpha',0.4);
% %            grid on;
%    %         legend({'Subspace (1/4))','gaussian','Subspace no scaling','Subspace (1/3)','Achlioptas', 'Householder'},'Interpreter','Latex');  
% 
%             figure;
%       
%       
% 
% 
% subplot(2,2,2);      
%       plot(xSub2,kdensSub2, 'b-');
% %      plot(xSub,kdensSub, 'b-');
%        title(['Kernel Density Plot for ',num2str(subspaces(subspaceIdx)),' projections.']);
% subplot(2,2,1);  
%        plot(xRP,kdensRP, 'r-');
% subplot(2,2,4);         
% %       plot(xSub3,kdensSub3, 'k-');
%        plot(xSRP,kdensSRP, 'm-');
% subplot(2,2,3);         
%        plot(xHSub,kdensHSub, 'c-');
%       
%        
%        
% 
% % %       plot(binCenter4,PCANHist, 'm-');
%       grid on;
%          legend({'Subspace ($1/6 (3-\sqrt{3}$))','gaussian','Subspace no scaling','Subspace (1/3 scaling)','Achlioptas', 'Householder'},'Interpreter','Latex');  
%          hold off;
% 

              
i=0         
for probNo=[2 3 5 6 7]
   i=i+1
   subplot(5,3,i);
    pr1=plot(subspaces, reshape(rpInt(probNo,1,:,1),1,[]),'r:');
hold on
pr2=plot(subspaces, reshape(rpStat(probNo,1,:,1),1,[]),'r.:');
pr3=plot(subspaces, reshape(rpInt(probNo,1,:,3),1,[]),'r:');

ps1=plot(subspaces, reshape(subspaceInt(probNo,1,:,1),1,[]),'b:');
ps2=plot(subspaces, reshape(subspaceStat(probNo,1,:,1),1,[]),'b+:');
ps3=plot(subspaces, reshape(subspaceInt(probNo,1,:,3),1,[]),'b:');

phs1=plot(subspaces, reshape(hsubspaceInt(probNo,1,:,1),1,[]),'g:');
phs2=plot(subspaces, reshape(hsubspaceStat(probNo,1,:,1),1,[]),'go:');
phs3=plot(subspaces, reshape(hsubspaceInt(probNo,1,:,3),1,[]),'g:');

psr1=plot(subspaces, reshape(srpInt(probNo,1,:,1),1,[]),'m:');
psr2=plot(subspaces, reshape(srpStat(probNo,1,:,1),1,[]),'mx:');
psr3=plot(subspaces, reshape(srpInt(probNo,1,:,3),1,[]),'m:');

%pp1=plot(subspaces, pcaInt(:,[1 3]),'m:');
%pp2=plot(subspaces, pcaInt(:,2),'mx:');

title([' dim (2500) p=' num2str(prob(probNo))]);
legend([pr2 ps2 phs2 psr2],'Gaussian','Subspace','Householder','Archiolptas');
ylim([0 2]);
hold off;
i=i+1
   subplot(5,3,i);
    
pr1=plot(subspaces, reshape(rpInt(probNo,dimCount,:,1),1,[]),'r:');
hold on
pr2=plot(subspaces, reshape(rpStat(probNo,dimCount,:,1),1,[]),'r.:');
pr3=plot(subspaces, reshape(rpInt(probNo,dimCount,:,3),1,[]),'r:');

ps1=plot(subspaces, reshape(subspaceInt(probNo,dimCount,:,1),1,[]),'b:');
ps2=plot(subspaces, reshape(subspaceStat(probNo,dimCount,:,1),1,[]),'b+:');
ps3=plot(subspaces, reshape(subspaceInt(probNo,dimCount,:,3),1,[]),'b:');

phs1=plot(subspaces, reshape(hsubspaceInt(probNo,dimCount,:,1),1,[]),'g:');
phs2=plot(subspaces, reshape(hsubspaceStat(probNo,dimCount,:,1),1,[]),'go:');
phs3=plot(subspaces, reshape(hsubspaceInt(probNo,dimCount,:,3),1,[]),'g:');

psr1=plot(subspaces, reshape(srpInt(probNo,dimCount,:,1),1,[]),'m:');
psr2=plot(subspaces, reshape(srpStat(probNo,dimCount,:,1),1,[]),'mx:');
psr3=plot(subspaces, reshape(srpInt(probNo,dimCount,:,3),1,[]),'m:');

%pp1=plot(subspaces, pcaInt(:,[1 3]),'m:');
%pp2=plot(subspaces, pcaInt(:,2),'mx:');

title(['dim (10000) p=' num2str(prob(probNo))]);
legend([pr2 ps2 phs2 psr2],'Gaussian','Subspace','Householder','Archiolptas');
ylim([0 2]);
hold off;
  
     
 i=i+1   
   subplot(5,3,i);    
    
pr1=plot(subspaces, reshape(mean(rpInt(probNo,:,:,1),2),1,[]),'r:');
hold on
pr2=plot(subspaces, reshape(mean(rpStat(probNo,:,:,1),2),1,[]),'r.:');
pr3=plot(subspaces, reshape(mean(rpInt(probNo,:,:,3),2),1,[]),'r:');

ps1=plot(subspaces, reshape(mean(subspaceInt(probNo,:,:,1),2),1,[]),'b:');
ps2=plot(subspaces, reshape(mean(subspaceStat(probNo,:,:,1),2),1,[]),'b+:');
ps3=plot(subspaces, reshape(mean(subspaceInt(probNo,:,:,3),2),1,[]),'b:');

phs1=plot(subspaces, reshape(mean(hsubspaceInt(probNo,:,:,1),2),1,[]),'g:');
phs2=plot(subspaces, reshape(mean(hsubspaceStat(probNo,:,:,1),2),1,[]),'go:');
phs3=plot(subspaces, reshape(mean(hsubspaceInt(probNo,:,:,3),2),1,[]),'g:');

psr1=plot(subspaces, reshape(mean(srpInt(probNo,:,:,1),2),1,[]),'m:');
psr2=plot(subspaces, reshape(mean(srpStat(probNo,:,:,1),2),1,[]),'mx:');
psr3=plot(subspaces, reshape(mean(srpInt(probNo,:,:,3),2),1,[]),'m:');

%pp1=plot(subspaces, pcaInt(:,[1 3]),'m:');
%pp2=plot(subspaces, pcaInt(:,2),'mx:');

title(['avged dim p=' num2str(prob(probNo))]);
legend([pr2 ps2 phs2 psr2],'Gaussian','Subspace','Householder','Archiolptas');
ylim([0 2]);
hold off;
end