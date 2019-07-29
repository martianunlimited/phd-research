%% Calculates the diversity measures between the base classifier and the saccade views 
% Table 7.5

dirnames={'alexnet', 'googlenet', 'resnet50','resnet152'}
obs=50000;
subspaces=[450 430 410 390 370 350 330 310 290 270 250 200 150];
directories=1:26;
kSize=size(subspaces,2);
dSize=size(directories,2);
classSize=1000;
corr2Score=zeros(1,10,classSize,dSize);

for z=1:4
   
   dirname=dirnames{z}
   baseFile=sprintf('%s/base/result.mat',dirname)
   load(baseFile,'results');
   baseRank=results(:,1);
%   for k=1:kSize
      subspace=450; %subspace=subspaces(k);
      ensSize=0;
      saccRank=zeros(50,classSize,dSize);      
      for j=1:dSize
         try
            directory=directories(j);
            
            filename=sprintf('%s/run_%d/result_%d.mat',dirname,directory,subspace);
            load(filename, 'results');
            for class=1:classSize
            saccRank(:,class,j)=results(solnInst(class,:),1);
            end
            ensSize=ensSize+1;
    
         catch
             
         end
      end
      for j=1:ensSize
        for class=1:classSize
        ensTops=sum((saccRank(:,class,[1:j])==1),3);
        corr2Score(z,1,class,j)=sum(ensTops.*(j-ensTops))/50/j/j;
        corr2Score(z,2,class,j)=sum(min(ensTops,j-ensTops))/(j-ceil(j/2))/50;
        avgAcc=sum(ensTops)/50/j;
        corr2Score(z,3,class,j)=avgAcc;
        corr2Score(z,4,class,j)=1 - ((sum(ensTops.*(j-ensTops))/j)/ (50*(j-1)*(avgAcc)*(1-avgAcc)));
        corr2Score(z,5,class,j)=var(ensTops/j);
        corr2Score(z,6,class,j)=1-var(ensTops/j)/(avgAcc*(1-avgAcc));
        corr2Score(z,7,class,j)=1-mean((j-ensTops).*(j-1-ensTops)/j/(j-1))/mean((j-ensTops)/j);
        end
        end
   end
%end
