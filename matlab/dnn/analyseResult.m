%% This routine generates a confusion matrix between a reference run and the 
% classification of a PseudoSaccade view
% confMat contains

%% Change the file below to the reference file
load('oldResult7/result_ref.mat','results')
 
refIs1=cell2mat(results(:,1))==1;
refIs2=cell2mat(results(:,1))==2;
refIs3=cell2mat(results(:,1))==3;
refIs4=cell2mat(results(:,1))==4;
refIs5=cell2mat(results(:,1))==5;
refIs6=cell2mat(results(:,1))>5;

% List of Projection dimensions to analyse
subspaces=[450 430 410 390 370 350 330 310 290 270 250 200 150];


%list of consecutive runs to analyse
directories=[2:24]
for j=1:size(directories,2)
     directory=directories(j)
 for k=1:size(subspaces,2)
     subspace=subspaces(k);
 
filename=sprintf('googlenet/run_%d/result_%d.mat',directory,subspace);

load(filename, 'results');
confMat=zeros(6,6);

confMat(1,1)=sum(results(refIs1,1)==1);
confMat(1,2)=sum(results(refIs1,1)==2);
confMat(1,3)=sum(results(refIs1,1)==3);
confMat(1,4)=sum((results(refIs1,1))==4);
confMat(1,5)=sum((results(refIs1,1))==5);
confMat(1,6)=sum((results(refIs1,1))>5);

confMat(2,1)=sum((results(refIs2,1))==1);
confMat(2,2)=sum((results(refIs2,1))==2);
confMat(2,3)=sum((results(refIs2,1))==3);
confMat(2,4)=sum((results(refIs2,1))==4);
confMat(2,5)=sum((results(refIs2,1))==5);
confMat(2,6)=sum((results(refIs2,1))>5);

confMat(3,1)=sum((results(refIs3,1))==1);
confMat(3,2)=sum((results(refIs3,1))==2);
confMat(3,3)=sum((results(refIs3,1))==3);
confMat(3,4)=sum((results(refIs3,1))==4);
confMat(3,5)=sum((results(refIs3,1))==5);
confMat(3,6)=sum((results(refIs3,1))>5);

confMat(4,1)=sum((results(refIs4,1))==1);
confMat(4,2)=sum((results(refIs4,1))==2);
confMat(4,3)=sum((results(refIs4,1))==3);
confMat(4,4)=sum((results(refIs4,1))==4);
confMat(4,5)=sum((results(refIs4,1))==5);
confMat(4,6)=sum((results(refIs4,1))>5);

confMat(5,1)=sum((results(refIs5,1))==1);
confMat(5,2)=sum((results(refIs5,1))==2);
confMat(5,3)=sum((results(refIs5,1))==3);
confMat(5,4)=sum((results(refIs5,1))==4);
confMat(5,5)=sum((results(refIs5,1))==5);
confMat(5,6)=sum((results(refIs5,1))>5);

confMat(6,1)=sum((results(refIs6,1))==1);
confMat(6,2)=sum((results(refIs6,1))==2);
confMat(6,3)=sum((results(refIs6,1))==3);
confMat(6,4)=sum((results(refIs6,1))==4);
confMat(6,5)=sum((results(refIs6,1))==5);
confMat(6,6)=sum((results(refIs6,1))>5);
disp(sprintf('dir=%d subspace=%d',directory,subspace));
disp(confMat);
 end
end
