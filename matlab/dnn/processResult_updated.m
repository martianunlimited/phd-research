modelPath = '../data/models/imagenet-googlenet-dag.mat' ;
lines = dataread('file', '../ILSVRC2012_validation_ground_truth.txt', '%s', 'delimiter', '\n', 'bufsize', 655350);
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

subspaces=[450 430 410 390 370 350 330 310 290 270 250 200 150];
noSubspace=size(subspaces,2);

directories=dir('run*');
noDir=size(directories,1);

records=zeros(6,noDir,noSubspace);
records2=zeros(6,noDir,noSubspace);
 for k=1:noSubspace
     subspace=subspaces(k)
best=sparse(50000,1000);

     for j=1:noDir
     directory=directories(j).name

    
filename=sprintf('%s/result_%d.mat',directory,subspace);
if exist(filename)==0 
    break;
end
load(filename, 'results');

results(19877,2)=1000;
results(19877,4)=1000;
results(19877,6)=1000;
results(19877,8)=1000;
results(19877,10)=1000;
best2=sparse([1:50000],results(:,2),results(:,3));
best=best+best2;
best2=sparse([1:50000],results(:,4),results(:,5));
best=best+best2;
best2=sparse([1:50000],results(:,6),results(:,7));
best=best+best2;
best2=sparse([1:50000],results(:,8),results(:,9));
best=best+best2;
best2=sparse([1:50000],results(:,10),results(:,11));
best=best+best2;
top1=sum(results(:,1)==1);
top2=sum(results(:,1)==2);
top3=sum(results(:,1)==3);
top4=sum(results(:,1)==4);
top5=sum(results(:,1)==5);
top6=sum(results(:,1)>5);

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
records(:,j,k)=tempRecord;
records2(1,j,k)=top1;
records2(2,j,k)=top2;
records2(3,j,k)=top3;
records2(4,j,k)=top4;
records2(5,j,k)=top5;
records2(6,j,k)=top6;

     end
 end
 
 
