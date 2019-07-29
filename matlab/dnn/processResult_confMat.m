%% processResults*.m are helper scripts to ensemble the Saccades using various methods
% results reported are from processResult_alexNet.m,
% processResults_GoogLeNet.m, processResults_r50.m and
% processResults_r152.m 
% This file prepares the confusion matrix for analysis with analyseResult
% and classAnalyzer

modelPath = 'data/models/imagenet-googlenet-dag.mat' ;
lines = dataread('file', 'ILSVRC2012_validation_ground_truth.txt', '%s', 'delimiter', '\n', 'bufsize', 655350);
obs=length(lines)
soln=zeros(1,obs);

solnMeta=load('meta.mat');
metaSize=size(solnMeta.synsets,1);



if ~exist(modelPath)
  mkdir(fileparts(modelPath)) ;
  urlwrite(...
  'http://www.vlfeat.org/matconvnet/models/imagenet-googlenet-dag.mat', ...
    modelPath) ;
end

net = dagnn.DagNN.loadobj(load(modelPath)) ;
metaMap=cell(metaSize,1);
for i=1:metaSize
   IndexC=strfind(net.meta.classes.name,solnMeta.synsets(i).WNID);
   if (isempty(find(not(cellfun('isempty',IndexC))))==1)
       metaMap{i}=-1;
   else
       metaMap{i}=find(not(cellfun('isempty',IndexC)));
   end
end

for i=1:obs
    soln(i)=str2num(lines{i});
end
soln=cell2mat(metaMap(soln'));

subspaces=[450  410  370 384 352 320 256];
ls run*;

noSubspace=size(subspaces,2);

ensSize=13;
records=zeros(1,6,ensSize,noSubspace);
records2=zeros(1,6,ensSize,noSubspace);
topSolution=zeros(obs,ensSize,noSubspace);
for zi=1:1
    directories=101:113;
    noDir=size(directories,2);

%directories=directories(randperm(noDir));
counter=0;
 for k=1:noSubspace
     
subspace=subspaces(k)
     
best=sparse(50000,1000);

     for j=1:ensSize
         counter=counter+1;
     try    
     directory=directories(j);

    
filename=sprintf('resnet50/run_%d/result_%d.mat',directory,subspace);

load(filename, 'results');



for zz=[5]
results(19877,zz,2)=1000;
results(19877,zz,4)=1000;
results(19877,zz,6)=1000;
results(19877,zz,8)=1000;
results(19877,zz,10)=1000;

    best2=sparse([1:50000],results(:,zz,2),results(:,zz,3));
best=best+best2;
best2=sparse([1:50000],results(:,zz,4),results(:,zz,5));
best=best+best2;
best2=sparse([1:50000],results(:,zz,6),results(:,zz,7));
best=best+best2;
best2=sparse([1:50000],results(:,zz,8),results(:,zz,9));
best=best+best2;
best2=sparse([1:50000],results(:,zz,10),results(:,zz,11));
best=best+best2;
end

top1=sum(results(:,11,1)==1);
top2=sum(results(:,11,1)==2);
top3=sum(results(:,11,1)==3);
top4=sum(results(:,11,1)==4);
top5=sum(results(:,11,1)==5);
top6=sum(results(:,11,1)>5);

myResult=zeros(50000,5);
for(i=1:50000)
    ranks=full(best(i,:));
    [sortX,sortI]=sort(ranks,'descend');
    myResult(i,1)=sortI(1);
    myResult(i,2)=sortI(2);
    myResult(i,3)=sortI(3);
    myResult(i,4)=sortI(4);
    myResult(i,5)=sortI(5);       
end
temp=repmat(soln,1,5);
temp2=(myResult==temp);
topSolution(:,j,k)=myResult(:,1);
tempRecord(1:5)=sum(temp2);
tempRecord(6)=49999-sum(sum(temp2));
tempRecord;
records(zi,:,j,k)=tempRecord;
records2(zi,1,j,k)=top1;
records2(zi,2,j,k)=top2;
records2(zi,3,j,k)=top3;
records2(zi,4,j,k)=top4;
records2(zi,5,j,k)=top5;
records2(zi,6,j,k)=top6;



     catch
     end
         
end
     
% filename=sprintf('oldResult4/result_%d.mat',subspace);
% load(filename, 'results');     
% results{19877,2}=1000;
% results{19877,4}=1000;
% results{19877,6}=1000;
% results{19877,8}=1000;
% results{19877,10}=1000;
% 
% results{19877,3}=single(0);
% results{19877,5}=single(0);
% results{19877,7}=single(0);
% results{19877,9}=single(0);
% results{19877,11}=single(0);

% best2=sparse([1:50000],cell2mat(results(:,2)),double(cell2mat(results(:,3))));
% best=best+best2;
% best2=sparse([1:50000],cell2mat(results(:,4)),double(cell2mat(results(:,5))));
% best=best+best2;
% best2=sparse([1:50000],cell2mat(results(:,6)),double(cell2mat(results(:,7))));
% best=best+best2;
% best2=sparse([1:50000],cell2mat(results(:,8)),double(cell2mat(results(:,9))));
% best=best+best2;
% best2=sparse([1:50000],cell2mat(results(:,10)),double(cell2mat(results(:,11))));
% best=best+best2;
% 
% myResult=zeros(50000,5);
% for(i=1:50000)
%     ranks=full(best(i,:));
%     [sortX,sortI]=sort(ranks,'descend');
%     myResult(i,1)=sortI(1);
%     myResult(i,2)=sortI(2);
%     myResult(i,3)=sortI(3);
%     myResult(i,4)=sortI(4);
%     myResult(i,5)=sortI(5);       
% end
% temp=repmat(soln,1,5);
% temp2=(myResult==temp);
% tempRecord(1:5)=sum(temp2);
% tempRecord(6)=49999-sum(sum(temp2));
% records(:,noDir+1,k)=tempRecord;
% 
% top1=sum(cell2mat(results(:,1))==1);
% top2=sum(cell2mat(results(:,1))==2);
% top3=sum(cell2mat(results(:,1))==3);
% top4=sum(cell2mat(results(:,1))==4);
% top5=sum(cell2mat(results(:,1))==5);
% top6=sum(cell2mat(results(:,1))>5);
% 
% records2(1,noDir+1,k)=top1;
% records2(2,noDir+1,k)=top2;
% records2(3,noDir+1,k)=top3;
% records2(4,noDir+1,k)=top4;
% records2(5,noDir+1,k)=top5;
% records2(6,noDir+1,k)=top6;
% 
% 
% filename=sprintf('oldResult3/result_%d.mat',subspace);
% load(filename, 'results');     
% results{19877,2}=1000;
% results{19877,4}=1000;
% results{19877,6}=1000;
% results{19877,8}=1000;
% results{19877,10}=1000;
% 
% results{19877,3}=single(0);
% results{19877,5}=single(0);
% results{19877,7}=single(0);
% results{19877,9}=single(0);
% results{19877,11}=single(0);
% 
% best2=sparse([1:50000],cell2mat(results(:,2)),double(cell2mat(results(:,3))));
% best=best+best2;
% best2=sparse([1:50000],cell2mat(results(:,4)),double(cell2mat(results(:,5))));
% best=best+best2;
% best2=sparse([1:50000],cell2mat(results(:,6)),double(cell2mat(results(:,7))));
% best=best+best2;
% best2=sparse([1:50000],cell2mat(results(:,8)),double(cell2mat(results(:,9))));
% best=best+best2;
% best2=sparse([1:50000],cell2mat(results(:,10)),double(cell2mat(results(:,11))));
% best=best+best2;
% 
% myResult=zeros(50000,5);
% for(i=1:50000)
%     ranks=full(best(i,:));
%     [sortX,sortI]=sort(ranks,'descend');
%     myResult(i,1)=sortI(1);
%     myResult(i,2)=sortI(2);
%     myResult(i,3)=sortI(3);
%     myResult(i,4)=sortI(4);
%     myResult(i,5)=sortI(5);       
% end
% temp=repmat(soln,1,5);
% temp2=(myResult==temp);
% tempRecord(1:5)=sum(temp2);
% tempRecord(6)=49999-sum(sum(temp2));
% records(:,noDir+2,k)=tempRecord;
% 
% top1=sum(cell2mat(results(:,1))==1);
% top2=sum(cell2mat(results(:,1))==2);
% top3=sum(cell2mat(results(:,1))==3);
% top4=sum(cell2mat(results(:,1))==4);
% top5=sum(cell2mat(results(:,1))==5);
% top6=sum(cell2mat(results(:,1))>5);
% 
% records2(1,noDir+2,k)=top1;
% records2(2,noDir+2,k)=top2;
% records2(3,noDir+2,k)=top3;
% records2(4,noDir+2,k)=top4;
% records2(5,noDir+2,k)=top5;
% records2(6,noDir+2,k)=top6;

 end
 

%  subspaces=[450 430 410 390 370 350 330 310 290 270 250 200 150];
% noSubspace=size(subspaces,2);
%  for k=1:noSubspace
%      subspace=subspaces(k)
%      best=sparse(50000,1000);
% 
%     
%     
% filename=sprintf('result_%d.mat',subspace);
% 
% load(filename, 'results');
% 
% results(19877,2)=1000;
% results(19877,4)=1000;
% results(19877,6)=1000;
% results(19877,8)=1000;
% results(19877,10)=1000;
% best2=sparse([1:50000],results(:,2),results(:,3));
% best=best+best2;
% best2=sparse([1:50000],results(:,4),results(:,5));
% best=best+best2;
% best2=sparse([1:50000],results(:,6),results(:,7));
% best=best+best2;
% best2=sparse([1:50000],results(:,8),results(:,9));
% best=best+best2;
% best2=sparse([1:50000],results(:,10),results(:,11));
% best=best+best2;
% top1=sum(results(:,1)==1);
% top2=sum(results(:,1)==2);
% top3=sum(results(:,1)==3);
% top4=sum(results(:,1)==4);
% top5=sum(results(:,1)==5);
% top6=sum(results(:,1)>5);
% 
% myResult=zeros(50000,5);
% for(i=1:50000)
%     ranks=full(best(i,:));
%     [sortX,sortI]=sort(ranks,'descend');
%     myResult(i,1)=sortI(1);
%     myResult(i,2)=sortI(2);
%     myResult(i,3)=sortI(3);
%     myResult(i,4)=sortI(4);
%     myResult(i,5)=sortI(5);       
% end
% temp=repmat(soln,1,5);
% temp2=(myResult==temp);
% tempRecord(1:5)=sum(temp2);
% tempRecord(6)=49999-sum(sum(temp2));
% tempRecord
% 
% end
records;
records2;
end

top1=reshape(sum(records(:,1,:,:),2),ensSize,noSubspace);
top3=reshape(sum(records(:,1:3,:,:),2),ensSize,noSubspace);
top5=reshape(sum(records(:,1:5,:,:),2),ensSize,noSubspace);

meanT1=(mean(top1,1)/49999 * 100)
meanT3=(mean(top3,1)/49999 * 100)
meanT5=(mean(top5,1)/49999 * 100)

stdT1=std(top1,1)/49999 * 100;
stdT3=std(top3,1)/49999 * 100;
stdT5=std(top5,1)/49999 * 100;
