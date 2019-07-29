%%% Norm preservation run with sparse dataset - Dorothea dataset
% 
% 1:  Setup the run and initialize the storage variables
% 2a: Sparse Achlioptas projection 
% 2b: Random Subspace projection 
% 2c: Random Subspace projection with Householder transfrom 
% 2d: Random Gaussian projection (non-orthornomalized)
% 3:  Calculate the statistics of the projected norms
% 4:  Visualize the data


%%% 1) Setup the run and initialize the variables
%%% variables
% files: data array to store the list of files for the runs
% fileCount: count the number of distinct files 
% fileRepeat: number of runs each files should be repeated

files=dir('../dataset/UCI/dorothea*.data');
fileCount=size(files,1);
fileRepeat=5;

% subspaces: list of subspace projections to run for, 
% subspaceCount: number of subspace projections in the list
% pairs: number of observations to be used for pairwise comparison

subspaces=[5 10 25 50 100 250:250:2500 2500:2500:25000 25000:5000:70000];
subspaceCount=size(subspaces,2);
pairs=100;

% normSub: data structure to store the calculated ||PX||/||X|| for subspace
%          projection
% normHSub: data structure to store the calculated ||PX||/||X|| for subspace
%           projection with householder transform applied
% normRP: data structure to store the calculated ||PX||/||X|| for random
%         gaussian projection
% normSRP: data structure to store the calculated ||PX||/||X|| for sparse
%          achlioptas projection
normSub=zeros(fileCount*fileRepeat,subspaceCount,pairs*(pairs-1)/2);
normHSub=zeros(fileCount*fileRepeat,subspaceCount,pairs*(pairs-1)/2);
normRP=zeros(fileCount*fileRepeat,subspaceCount,pairs*(pairs-1)/2);
normSRP=zeros(fileCount*fileRepeat,subspaceCount,pairs*(pairs-1)/2);

% subspaceStat: data structure to aggregate the statistics for subspace
%               projection
% hsubspaceStat: data structure to aggregate the statistics for subspace
%                projection with householder transform applied
% rpStat: data structure to aggregate the statistics for random gaussian
%         projection
% srpStat: data structure to aggregate the statistics for sparse achlioptas
%          projection
% by file and subspace.
% field_1: average
% field_2: variance
% field_3: minimum
% field_4: max
subspaceStat=zeros(fileCount*fileRepeat,subspaceCount,4);
hsubspaceStat=zeros(fileCount*fileRepeat,subspaceCount,4);
rpStat=zeros(fileCount*fileRepeat,subspaceCount,4);
srpStat=zeros(fileCount*fileRepeat,subspaceCount,4);

% data structure for aggregation (storing the percentile observation) by
% file and subspace, with 3 fields
% subspaceInt:
% hsubspaceInt:
% rpInt:
% srpInt: 
% field_1: 5th percentile observation
% field_2: 50th percentile observation
% field_3: 95th percentil observation

subspaceInt=zeros(fileCount*fileRepeat,subspaceCount,3);
hsubspaceInt=zeros(fileCount*fileRepeat,subspaceCount,3);
rpInt=zeros(fileCount*fileRepeat,subspaceCount,3);
srpInt=zeros(fileCount*fileRepeat,subspaceCount,3);

% subspaceRunTime: data structure for 100x99/2 (100 choose 2) norm
% comparisons, by file and subspace. 
% hsubspaceRunTime
% rpRunTime:
% srpRunTime: 
subspaceRunTime=zeros(fileCount*fileRepeat,subspaceCount);
hsubspaceRunTime=zeros(fileCount*fileRepeat,subspaceCount);
rpRunTime=zeros(fileCount*fileRepeat,subspaceCount);
srpRunTime=zeros(fileCount*fileRepeat,subspaceCount);

% scaleFactor: calculates and stores the value of "c", regularity constnant
scaleFactor=zeros(fileCount*fileRepeat,1);


%%% Begining of the loop.
% For each iteration (5)
%   For each file
%      for each subspace

% fileIndex: keeps track of the individual runs
% fileRun: keeps track of the number of times the dataset have been run
% fileNo: keeps track of which dataset is being run

fileIndex=0;
for fileRuns=1:fileRepeat
    for fileNo=1:fileCount
        fileName=files(fileNo).name
        fileIndex=fileIndex+1

% Read in the file by line, the data file stores the index of the "1s" and
% each record is split by line
        
% lines: stores the file into the memory, seperated by newline         
% obs: counts the number of records
% S: stores the data as a sparse binary matrix
        lines = dataread('file', ['../dataset/UCI/' fileName], '%s', 'delimiter', '\n', 'bufsize', 655350);
        obs=length(lines)
        S=sparse(100000,obs);

% for each of the record, obtain the index, and update the data matrix 
        for i=1:obs
            index=sscanf(char(lines(i)),'%d');
            S(index,i)=1;
        end

% trim the data matrix to remove columns that are all 0s (i.e. columns with 0 variance 
% note: we could be more robust to remove the coulms that are all 1s, but none exist 
% in this dataset), 
% dim: stores the number of columns remaining in S
        S=S(sum(S,2)>0,:);
        dim=size(S,1);

        u=(1-(1-(sum(S,2)/obs)).^2); 
        v=householder_sv(dim);
        n=householderNormal(u,v);
% generate the normal vector for householder reflection... 
% note: this code is legacy code, after going through the theory, we've found 
% an efficient reflection that works efficiently all {0,1}^d vectors, we
% kept it in here for ease of experimentation and to try different
% reflections

%
%        tic
%            v=ones(1,dim)*sqrt(1/dim);
%        toc
%        %alpha=full(v*S);

% calculate the value for c (regularity constant).. leveraging on the fact
% that ||x||_inf = 1, and ||x||_2 = sqrt(s) with s being the number of
% non-zero observation, we can calculate c=dim / s
%
% number of samples to use to estimate c, use obs to use all observations
% and obtain the exact value of c.
        minSpars=dim;
        sig=S'*S;
        numSamp=obs;
        for i=1:numSamp
            for j=i+1:numSamp
                sampSpars=sig(i,i)-2*sig(i,j)+sig(j,j);
                if minSpars>sampSpars
                    minSpars=sampSpars;
                end
            end
        end
        
        scale=dim/minSpars;
        scaleFactor(fileIndex)=scale;
        
% Loop through the subspaces
% subspace: stores the number of subspaces for current run
% randIdx: stores the observations to be used for the runs, we choose
% "pairs" observations from "obs" observations
        
        for k=1:size(subspaces,2)    
            subspace=subspaces(k)
            randIdx=randperm(obs,pairs);

% 2a) Sparse Achlioptas Projection run
% We only run for subspace<3000 as we run into memory issues and
% system instability when subspace>3000
% SR stores the random projeciton matrix
% SP stores the projected data (i.e. PX)
% uv stores the difference of the vectors
% uvP stores the difference of the projected vector
% note: we use full() to convert the sparse matrix into a normal matrix as
% sum is inefficient when runing on a sparse matrix.
% for subspace >= 3000, we use the result of our last run (subspace=2750)

            if subspace<1000
                l=0;
                tic;
                SR=randproj_3(subspace,dim);
                SP=SR*S;
                for i=1:pairs
                    for j=i+1:pairs
                        l=l+1;
                        uv=full(S(:,randIdx(i))-S(:,randIdx(j)));
                        uvP=full(SP(:,randIdx(i))-SP(:,randIdx(j)));
                        normSRP(fileIndex,k,l)=sqrt(sum((uvP).^2)/ sum(uv.^2)/subspace);
                    end
                end
                srpRunTime(fileIndex,k)=toc;
            else
                normSRP(fileIndex,k,:)=normSRP(fileIndex,k-1,:);
                srpRunTime(fileIndex,k)=srpRunTime(fileIndex,k-1);
            end
            
            l=0;

% 2b) Random Subspace projection
% P contains the indices of the columns selected by the random subspace
% projection, note: due to inefficiencies in the implementation of accessing the index of a 
% sparse matrices, the indices should be sorted to improve the runtime
% SP stores the projected data (i.e. PX)
% uv stores the difference of the vectors
% uvP stores the difference of the projected vector
                                    
            tic;
            P=randperm(dim,ceil(subspace));
            P=sort(P);
            SP=S(P,:);

            for i=1:pairs
                for j=i+1:pairs
                    l=l+1;
                    uv=full(S(:,randIdx(i))-S(:,randIdx(j)));
                    uvP=full(SP(:,randIdx(i))-SP(:,randIdx(j)));   
                    normSub(fileIndex,k,l)=sqrt(sum(uvP.^2) / sum(uv.^2)*dim/ceil(subspace));
                end
            end
            subspaceRunTime(fileIndex,k)=toc;
  
% 2c) Random Subspace projection with Householder transform
% P contains the indices of the columns selected by the random subspace projection
% we use the same indices from 2b, the runtime cost of generating P is negligible 
% SP stores the projected data (i.e. PX)
% uv stores the difference of the vectors
% uvP stores the difference of the projected vector            
% alpha2 stores the value of alpha (ie v^T X, with X=(u-v)), used in the legacy 
% implementation, since v=ones(d) /sqrt(d), we have a more efficient way of calculating uvP
% uvS stores the number of non-zero entries
% ||PH'(uvP)|| =  || PH abs(uvP) || =  

% we also leverage the fact that norm(uv) = norm (abs(uv)) in order to
% obtain an efficient H that works on the paired difference. 

            l=0;
            tic;
            % P=randperm(dim,ceil(subspace*scale2));
            P=sort(P);
            SP=S(P,:);
            for i=1:pairs
                for j=i+1:pairs
                    l=l+1;
                    uv=full(abs(S(:,randIdx(i))-S(:,randIdx(j))));
%                    uvP=full(abs(SP(:,randIdx(i))-SP(:,randIdx(j))));
%                    uvS=sum(uv);
%                    uvP=uvP-2*uvS/dim;
                    u=S(:,randIdx(i));
                    v=S(:,randIdx(j));
                    uH=u - n*(u'*n);
                    vH=v - n*(v'*n);
                    uHP=uH(P);
                    vHP=vH(P);
                    uvHP=uHP-vHP;

                    %alpha2=alpha(randIdx(i))-alpha(randIdx(j));
                    %beta=v(P)*uv(P);
                    normHSub(fileIndex,k,l)=sqrt((sum(uvHP.^2)) / sum(uv.^2)*dim/ceil(subspace));
                end
            end
            hsubspaceRunTime(fileIndex,k)=toc;
            

% 2d) Random Gaussian Projection run
% We only run for subspace<3000 as we run into memory issues and
% system instability when subspace>3000
% R stores the random projeciton matrix
% SP stores the projected data (i.e. PX)
% uv stores the difference of the vectors
% uvP stores the difference of the projected vector
% note: we use full() to convert the sparse matrix into a normal matrix as
% sum is inefficient when runing on a sparse matrix.
% for subspace >= 3000, we use the result of our last run (subspace=2750)            
            
            
            if subspace < 1000
                l=0;
                tic;
                R=randproj(subspace,dim);
                SP=R*S;
                for i=1:pairs
                    for j=i+1:pairs
                        l=l+1;
                        uv=full(S(:,randIdx(i))-S(:,randIdx(j)));
                        uvP=full(SP(:,randIdx(i))-SP(:,randIdx(j)));
                        normRP(fileIndex,k,l)=sqrt(sum(uvP.^2)/ sum(uv.^2)/subspace);
                    end
                end
                rpRunTime(fileIndex,k)=toc;
            else
                normRP(fileIndex,k,:)=normRP(fileIndex,k-1,:);
                rpRunTime(fileIndex,k)=rpRunTime(fileIndex,k-1);
            end
            
% 3:  Calculate the statistics of the projected norms
            
%%% We calculate the statistics of the runs and obtain the mean, variance, minimum, maximum, 5th, 50th and 95th percentile observation for each of the runs     
            
            subspaceStat(fileIndex,k,:)=[mean(normSub(fileIndex,k,:)) var(normSub(fileIndex,k,:)) min(normSub(fileIndex,k,:)) max(normSub(fileIndex,k,:))];
            hsubspaceStat(fileIndex,k,:)=[mean(normHSub(fileIndex,k,:)) var(normHSub(fileIndex,k,:)) min(normHSub(fileIndex,k,:)) max(normHSub(fileIndex,k,:))];
            rpStat(fileIndex,k,:)=[mean(normRP(fileIndex,k,:)) var(normRP(fileIndex,k,:)) min(normRP(fileIndex,k,:)) max(normRP(fileIndex,k,:))];
            srpStat(fileIndex,k,:)=[mean(normSRP(fileIndex,k,:)) var(normSRP(fileIndex,k,:)) min(normSRP(fileIndex,k,:)) max(normSRP(fileIndex,k,:))];
            
            subspaceInt(fileIndex,k,:)=prctile(normSub(fileIndex,k,:),[5 50 95]);
            hsubspaceInt(fileIndex,k,:)=prctile(normHSub(fileIndex,k,:),[5 50 95]);
            rpInt(fileIndex,k,:)=prctile(normRP(fileIndex,k,:),[5 50 95]);
            srpInt(fileIndex,k,:)=prctile(normSRP(fileIndex,k,:),[5 50 95]);

        end
    end
end


%%% We calculate the averages of each of the statistics
avgRPStat=mean(rpStat,1);
avgSRPStat=mean(srpStat,1);
avgSubspaceStat=mean(subspaceStat,1);
avgHSubspaceStat=mean(hsubspaceStat,1);

avgRPInt=mean(rpInt,1);
avgSRPInt=mean(srpInt,1);
avgSubspaceInt=mean(subspaceInt,1);
avgHSubspaceInt=mean(hsubspaceInt,1);

% 4:  Visualization of the statistics
% Figure 1 is a graph displaying the error intervals on a single graph 

figure;
pr1=plot(subspaces, avgRPInt(1,:,[1]),'r.:');
hold on
pr2=plot(subspaces, avgRPStat(1,:,1),'r.:');
pr3=plot(subspaces, avgRPInt(1,:,[3]),'r.:');

ps1=plot(subspaces, avgSubspaceInt(1,:,[1]),'b+:');
ps2=plot(subspaces, avgSubspaceStat(1,:,1),'b+:');
ps3=plot(subspaces, avgSubspaceInt(1,:,[3]),'b+:');
  
psr1=plot(subspaces, avgSRPInt(1,:,[1]),'mx:');
psr2=plot(subspaces, avgSRPStat(1,:,1),'mx:');
psr3=plot(subspaces, avgSRPInt(1,:,[3]),'mx:');

phs1=plot(subspaces, avgHSubspaceInt(1,:,[1]),'g:o');
phs2=plot(subspaces, avgHSubspaceStat(1,:,1),'go:');
phs3=plot(subspaces, avgHSubspaceInt(1,:,[3]),'g:o');
title('Sparse: 95% bounds for various projections');
legend([pr2 ps2 phs2 psr2],'Gaussian','Subspace','Householder','Achlioptas');
hold off;


% Figure 2 is a graph displaying the error intervals on 4 different subgraph 
figure;
subplot(2,2,1);
pr1=plot(subspaces, avgRPInt(1,:,[1]),'r:s');
hold on
pr2=plot(subspaces, avgRPStat(1,:,1),'k:s');
pr3=plot(subspaces, avgRPInt(1,:,[3]),'r:s');
hold off

subplot(2,2,2);
ps1=plot(subspaces, avgSubspaceInt(1,:,[1]),'b:+');
hold on;
ps2=plot(subspaces, avgSubspaceStat(1,:,1),'k+:');
ps3=plot(subspaces, avgSubspaceInt(1,:,[3]),'b:+');
hold off;

subplot(2,2,3);
phs1=plot(subspaces, avgHSubspaceInt(1,:,[1]),'g:o');
hold on;
phs2=plot(subspaces, avgHSubspaceStat(1,:,1),'ko:');
phs3=plot(subspaces, avgHSubspaceInt(1,:,[3]),'g:o');
hold off;

subplot(2,2,4);
psr1=plot(subspaces, avgSRPInt(1,:,[1]),'m:x');
hold on;
psr2=plot(subspaces, avgSRPStat(1,:,1),'kx:');
psr3=plot(subspaces, avgSRPInt(1,:,[3]),'mx:');
hold off;
title('Sparse: 95% bounds for various projections');
legend([pr2 ps2 phs2 psr2],'Gaussian','Subspace','Householder','Achlioptas');
hold off;


% Figure 2 is a graph displaying the ||PX||/||X|| for k = 1250 projections

figure;
     centers=0.49:0.01:1.51;
     normHSubT=reshape(permute(normHSub,[2 1 3]),1,size(subspaces,2),[]);
     normSubT=reshape(permute(normSub,[2 1 3]),1,size(subspaces,2),[]);

     normRPT=reshape(permute(normRP,[2 1 3]),1,size(subspaces,2),[]);
     normSRPT=reshape(permute(normSRP,[2 1 3]),1,size(subspaces,2),[]);
     
      [PXNHist, binCenter1]=hist(reshape(normSubT(1,10,:),1,[]),centers);
      [RXNHist, binCenter2]=hist(reshape(normRPT(1,10,:),1,[]),centers);
      [HPXNHist, binCenter3]=hist(reshape(normHSubT(1,10,:),1,[]),centers);
      [SRXNHist, binCenter5]=hist(reshape(normSRPT(1,10,:),1,[]),centers);
         ps1=plot(binCenter1,PXNHist, 'b-');
         hold on;
         title(['Sparse: Number of projections ',num2str(subspaces(10))]);
         pr1=plot(binCenter2,RXNHist, 'r-');
         phs1=plot(binCenter3,HPXNHist, 'g-');
         psr1=plot(binCenter5,SRXNHist, 'm-');
         grid on;
         legend([pr1 ps1 phs1 psr1],'Gaussian','Subspace','Householder','Achlioptas');
         hold off;
         figure

         
% Figure 4, displays the average run time for each of the projection
% methods (to project (pairs*pairs-1)/2 comparisons)
         
avgRPRunTime=mean(rpRunTime,1)
avgSRPRunTime=mean(srpRunTime,1)
avgSubspaceRunTime=mean(subspaceRunTime,1)
avgHSubspaceRunTime=mean(hsubspaceRunTime,1)

figure;

pr1=plot(subspaces, avgRPRunTime,'k-+');
hold on
title(['Runtime vs Projections ']);
ps1=plot(subspaces, avgSubspaceRunTime,'k-o');
%phs1=plot(subspaces, avgHSubspace2RunTime,'k-');
phss1=plot(subspaces, avgHSubspaceRunTime,'k-h');
psr1=plot(subspaces, avgSRPRunTime,'k-*');
         grid on;
legend([pr1 ps1 psr1 phss1],'Gaussian','Subspace','Achlioptas', 'Householder');
         hold off;

         
         
    
%%% Legacy code for debugging, unused...          
         
         
         
%   subspaceIdx=10;
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

%            figure;
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


figure;
pr1=plot(subspaces, avgRPInt(1,:,[1]),'r:s');
hold on
pr2=plot(subspaces, avgRPStat(1,:,1),'k:s');
pr3=plot(subspaces, avgRPInt(1,:,[3]),'r:s');

ps1=plot(subspaces, avgSubspaceInt(1,:,[1]),'b:+');
ps2=plot(subspaces, avgSubspaceStat(1,:,1),'k+:');
ps3=plot(subspaces, avgSubspaceInt(1,:,[3]),'b:+');
phs1=plot(subspaces, avgHSubspaceInt(1,:,[1]),'g:o');

phs2=plot(subspaces, avgHSubspaceStat(1,:,1),'ko:');
phs3=plot(subspaces, avgHSubspaceInt(1,:,[3]),'g:o');
psr1=plot(subspaces, avgSRPInt(1,:,[1]),'m:x');
psr2=plot(subspaces, avgSRPStat(1,:,1),'kx:');
psr3=plot(subspaces, avgSRPInt(1,:,[3]),'mx:');
legend([pr1 ps1 psr1 phs1],'Gaussian','Subspace','Achlioptas', 'Householder');
title('95% bounds for various projections');
xlim([0 600]);
ylim([0 2]);
xlabel('Projection Dimension(k)');
hold off;
    
         
         