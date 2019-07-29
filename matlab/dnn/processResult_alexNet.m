%% processResults*.m are helper scripts to ensemble the Saccades using various methods
% results reported are from processResult_Alexnet.m,
% processResults_GoogLeNet.m, processResults_r50.m and
% processResults_r152.m 
% This file ensembles the results using simple majority vote on AlexNet

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

subspaces=[450 410 370 384 256];
ls run*;

noSubspace=size(subspaces,2);
zCount=1
ensSize=9;
records=zeros(zCount,6,ensSize,noSubspace);
records2=zeros(zCount,6,ensSize,noSubspace);
for zi=1:zCount
    directories=101:109;
    noDir=size(directories,2);

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

results(19877,11,2)=1000;
results(19877,11,4)=1000;
results(19877,11,6)=1000;
results(19877,11,8)=1000;
results(19877,11,10)=1000;

best2=sparse([1:50000],results(:,11,2),results(:,11,3));
best=best+best2;
best2=sparse([1:50000],results(:,11,4),results(:,11,5));
best=best+best2;
best2=sparse([1:50000],results(:,11,6),results(:,11,7));
best=best+best2;
best2=sparse([1:50000],results(:,11,8),results(:,11,9));
best=best+best2;
best2=sparse([1:50000],results(:,11,10),results(:,11,11));
best=best+best2;
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
     
 end
 

records;
records2;
end

top1=reshape(sum(records(:,1,:,:),2),zCount,ensSize,noSubspace);
top3=reshape(sum(records(:,1:3,:,:),2),zCount,ensSize,noSubspace);
top5=reshape(sum(records(:,1:5,:,:),2),zCount,ensSize,noSubspace);

meanT1=(mean(top1,1)/49999 * 100)
meanT3=(mean(top3,1)/49999 * 100)
meanT5=(mean(top5,1)/49999 * 100)

stdT1=std(top1,1)/49999 * 100;
stdT3=std(top3,1)/49999 * 100;
stdT5=std(top5,1)/49999 * 100;
