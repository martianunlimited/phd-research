
%% Set this to 1 if you need to run the base classifier (on the full views)
runBase=0;

run matlab/vl_setupnn
modelPath = 'data/models/imagenet-matconvnet-alex.mat' ;
lines = dataread('file', 'ILSVRC2012_validation_ground_truth.txt', '%s', 'delimiter', '\n', 'bufsize', 655350);
obs=length(lines)
soln=zeros(1,obs);

solnMeta=load('meta.mat');
metaSize=size(solnMeta.synsets,1);



if ~exist(modelPath)
  mkdir(fileparts(modelPath)) ;
  urlwrite(...
  'http://www.vlfeat.org/matconvnet/models/imagenet-matconvnet-alex.mat', ...
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

files=dir('/Scratch/jsl18/*.JPEG'); 
fileCount=size(files,1);
if (runBase==1)
results=zeros(50000,12); tic; 
for fileNo=1:50000
    try
        filename=sprintf('/Scratch/jsl18/ILSVRC2012_val_%08d.JPEG',fileNo);
        im = imread(filename) ; ImSizes=size(im); rgb=size(im,3); 
        if (rgb==1)
            im=cat(3,im,im,im);
        end
        im2=im;
        
        im_ = single(im); 
        im_ = imresize(im_,net.meta.normalization.imageSize(1:2)) ;

        im_ = im_ - net.meta.normalization.averageImage ;
        net.eval({'data', im_}) ;

        scores = squeeze(gather(net.vars(end).value)) ;

        [sortedX,sortingIndices] = sort(scores,'descend');
        bestScore=sortedX(1); best=sortingIndices(1);
        thisSoln=soln(fileNo);
        bestRank=find(sortingIndices==metaMap{thisSoln}); 
        [bestScore,best] = max(scores); 

        % Store output, 
        % column 1 is rank of the true label
        % column 2,4,6,8,10 is the label of the top 1-5 prediction
        % column 3,5,7,9,11 are the scores of said predictions
        % column 12 is a map for the true label (used later in python) 
        results(fileNo,1)=bestRank; results(fileNo,2)=best;
        results(fileNo,3)=bestScore; results(fileNo,4)=sortingIndices(2);
        results(fileNo,5)=sortedX(2);
        results(fileNo,6)=sortingIndices(3);
        results(fileNo,7)=sortedX(3);
        results(fileNo,8)=sortingIndices(4);
        results(fileNo,9)=sortedX(4);
        results(fileNo,10)=sortingIndices(5);
        results(fileNo,11)=sortedX(5);
        results(fileNo,12)=metaMap{thisSoln};

       

%        bestOri=best; %        bestOriScore=bestScore; %
bestOriRank=bestRank;
    catch
        continue
    end
end
        resultFile=sprintf('result_ref.mat'); save(resultFile,'results');
toc
end

% classify using the PseudoSaccade views,
% setup the projection dimensions
        subspaces=[450 430 410 390 370 350 330 310 290 270 250 200 150];
% obtain the last `sucessful' run
        load('alexnet/dirCount.mat','dirCount');
        dirCount=dirCount+1;
        dirName=sprintf('alexnet/run_%d',dirCount);
        mkdir(dirName);
% Loop though each of the projection dimensions
        for k=1:size(subspaces,2)
            subspace=subspaces(k);
        sprintf('Subspaced Run subspace=%d ',subspace );
 tic
 results=zeros(50000,11);
for fileNo=1:50000
    try
        filename=sprintf('/Scratch/jsl18/ILSVRC2012_val_%08d.JPEG',fileNo);
        im = imread(filename) ;
        ImSizes=size(im);
        rgb=size(im,3);
        if (rgb==1)
            im=cat(3,im,im,im);
        end
       
        maxX=min(subspace,ImSizes(1));
        maxY=min(subspace,ImSizes(2));
        im2=zeros(maxX,maxY,3);
        im2 = im(sort(randperm(ImSizes(1),maxX)),sort(randperm(ImSizes(2),maxY)),:);
        im_ = single(im2) ; % note: 255 range
        im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;

        im_(:,:,1) = im_(:,:,1) - net.meta.normalization.averageImage(1) ;
        im_(:,:,2) = im_(:,:,2) - net.meta.normalization.averageImage(2) ;
        im_(:,:,3) = im_(:,:,3) - net.meta.normalization.averageImage(3) ;

        net.eval({'input', im_}) ;

        scores = squeeze(gather(net.vars(end).value)) ;

        [sortedX,sortingIndices] = sort(scores,'descend');
        bestScore=sortedX(1); best=sortingIndices(1);
        
        thisSoln=soln(fileNo);
        bestRank=find(sortingIndices==metaMap{thisSoln});
        % Store output, 
        % column 1 is rank of the true label
        % column 2,4,6,8,10 is the label of the top 1-5 prediction
        % column 3,5,7,9,11 are the scores of said predictions
        % column 12 is a map for the true label (used later in python) 
        results(fileNo,1)=bestRank;
        results(fileNo,2)=best;
        results(fileNo,3)=bestScore;
        results(fileNo,4)=sortingIndices(2);
        results(fileNo,5)=sortedX(2);
        results(fileNo,6)=sortingIndices(3);
        results(fileNo,7)=sortedX(3);
        results(fileNo,8)=sortingIndices(4);
        results(fileNo,9)=sortedX(4);
        results(fileNo,10)=sortingIndices(5);
        results(fileNo,11)=sortedX(5);


    catch
        continue
    end

end

        resultFile=sprintf('alexnet/run_%d/result_%d.mat',dirCount,subspace);
        save(resultFile,'results');
        toc
        end
save('alexnet/dirCount.mat','dirCount');       
