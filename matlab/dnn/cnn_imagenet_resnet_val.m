%CNN_IMAGENET_GOOGLENET  Demonstrates how to use GoogLeNet

%run matlab/vl_setupnn
modelPath = 'data/models/imagenet-resnet-50-dag.mat' ;
lines = dataread('file', 'ILSVRC2012_validation_ground_truth.txt', '%s', 'delimiter', '\n', 'bufsize', 655350);
obs=length(lines)
soln=zeros(1,obs);

solnMeta=load('meta.mat');
metaSize=size(solnMeta.synsets,1);



if ~exist(modelPath)
  mkdir(fileparts(modelPath)) ;
  urlwrite(...
  'http://www.vlfeat.org/matconvnet/models/imagenet-resnet-50-dag.mat', ...
    modelPath) ;
end

net = dagnn.DagNN.loadobj(load(modelPath)) ;
net.mode='test';
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


%files=dir('testImage/*.png');
files=dir('/Scratch/jsl18/*.JPEG'); 
fileCount=size(files,1);
results=zeros(50000,12); tic; 
% for fileNo=1:50000
%     try
%         filename=sprintf('/Scratch/jsl18/ILSVRC2012_val_%08d.JPEG',fileNo);
%         im = imread(filename) ; ImSizes=size(im); rgb=size(im,3); 
%         if (rgb==1)
%             im=cat(3,im,im,im);
%         end
%         im2=im;
% %tic; %im2 =im(sort(randperm(ImSizes(1),net.meta.normalization.imageSize(1))),sort(randperm(ImSizes(2),net.meta.normalization.imageSize(2))),:);
% %im_ = single(im2) ; % note: 255 range
%         
%         im_ = single(im); 
%         im_ = imresize(im_,net.meta.normalization.imageSize(1:2)) ;
% 
%         im_ = im_ - net.meta.normalization.averageImage ;
%         net.eval({'data', im_}) ;
% 
% % show the classification result
%         scores = squeeze(gather(net.vars(end).value)) ;
% 
%         [sortedX,sortingIndices] = sort(scores,'descend');
%         bestScore=sortedX(1); best=sortingIndices(1);
%         thisSoln=soln(fileNo);
%         bestRank=find(sortingIndices==metaMap{thisSoln}); 
%         [bestScore,best] = max(scores); 
%         %figure(1) ; clf ; imagesc(im2) ;
%         %title(sprintf('%s (%d), score %.3f',...
%         %net.meta.classes.description{best}, best, bestScore)) ;
%         results(fileNo,1)=bestRank; results(fileNo,2)=best;
%         results(fileNo,3)=bestScore; results(fileNo,4)=sortingIndices(2);
%         results(fileNo,5)=sortedX(2);
%         results(fileNo,6)=sortingIndices(3);
%         results(fileNo,7)=sortedX(3);
%         results(fileNo,8)=sortingIndices(4);
%         results(fileNo,9)=sortedX(4);
%         results(fileNo,10)=sortingIndices(5);
%         results(fileNo,11)=sortedX(5);
%         results(fileNo,12)=metaMap{thisSoln};
% 
%        
% 
% %        bestOri=best; %        bestOriScore=bestScore; %
% bestOriRank=bestRank;
%     catch
%         continue
%     end
% end
%         resultFile=sprintf('result_ref.mat'); save(resultFile,'results');
% toc

%for i=1:5 
    %sprintf('%s (%d), score %.3f',...net.meta.classes.description{sortingIndices(i)}, sortingIndices(i),sortedX(i)) %end
              subspaces=[450 430 410 390 370 350 330 310 290 270 250 200 150];
        %subspaces=[200 150 100];
        load('resnet50/dirCount.mat','dirCount');
        dirCount=dirCount+1;
        dirName=sprintf('resnet50/run_%d',dirCount);
        mkdir(dirName);
    
   % subspaces=[430 350 300 250 200 150 100];
        %subspaces=[200 150 100];
        
        for k=1:size(subspaces,2)
            subspace=subspaces(k);
%toc;
%tic;
        sprintf('Subspaced Run subspace=%d ',subspace );
 tic
 results=zeros(50000,11);
for fileNo=1:50000
    try
        rng(12382+dirCount);
        filename=sprintf('/Scratch/jsl18/ILSVRC2012_val_%08d.JPEG',fileNo);
        im = imread(filename) ;
        ImSizes=size(im);
        rgb=size(im,3);
        if (rgb==1)
            im=cat(3,im,im,im);
        end
       if mod(fileNo,5000)==0
           fileNo
       end
        maxX=min(subspace,ImSizes(1));
        maxY=min(subspace,ImSizes(2));
        im2=zeros(maxX,maxY,3);
        im2 = im(sort(randperm(ImSizes(1),maxX)),sort(randperm(ImSizes(2),maxY)),:);
%im2(:,:,1) = im(sort(randperm(ImSizes(1),maxX)),sort(randperm(ImSizes(2),maxY)),1);
%im2(:,:,2) = im(sort(randperm(ImSizes(1),maxX)),sort(randperm(ImSizes(2),maxY)),2);
%im2(:,:,3) = im(sort(randperm(ImSizes(1),maxX)),sort(randperm(ImSizes(2),maxY)),3);        
        im_ = single(im2) ; % note: 255 range
        im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;

        im_(:,:,1) = im_(:,:,1) - net.meta.normalization.averageImage(1) ;
        im_(:,:,2) = im_(:,:,2) - net.meta.normalization.averageImage(2) ;
        im_(:,:,3) = im_(:,:,3) - net.meta.normalization.averageImage(3) ;
        net.eval({'data', im_}) ;

% show the classification result
        scores = squeeze(gather(net.vars(end).value)) ;

        [sortedX,sortingIndices] = sort(scores,'descend');
        bestScore=sortedX(1); best=sortingIndices(1);
        
        thisSoln=soln(fileNo);
        bestRank=find(sortingIndices==metaMap{thisSoln});
%[bestScore, best] = max(scores) ;
%figure(2) ; clf ; imagesc(im2) ;
%title(sprintf('Subspaced %s (%d), score %.3f',...
%net.meta.classes.description{best}, best, bestScore)) ;
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

%for i=1:5
%sprintf('%s (%d), score %.3f',...
%net.meta.classes.description{sortingIndices(i)}, sortingIndices(i), sortedX(i))
%end
%toc;

end
        resultFile=sprintf('resnet50/run_%d/result_%d.mat',dirCount,subspace);
        save(resultFile,'results');
        toc
        end
save('resnet50/dirCount.mat','dirCount');   

