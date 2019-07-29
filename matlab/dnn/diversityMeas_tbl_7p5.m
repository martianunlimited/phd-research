%% Calculates the diversity measures between the base classifier and the saccade views 
% Table 7.5 and 7.6

dirnames={'alexnet', 'googlenet', 'resnet50','resnet152'}
obs=50000;
subspaces=[450 430 410 390 370 350 330 310 290 270 250 200 150];
directories=1:26;
kSize=size(subspaces,2);
dSize=size(directories,2);
corr2Score=zeros(4,10,dSize,kSize);

for z=1:4
   
   dirname=dirnames{z}
   baseFile=sprintf('%s/base/result.mat',dirname)
   load(baseFile,'results');
   baseRank=results(:,1);
   for k=1:kSize
      subspace=subspaces(k);
      ensSize=0;
      saccRank=zeros(obs,dSize);      
      for j=1:dSize
         directory=directories(j);
         try
         filename=sprintf('%s/run_%d/result_%d.mat',dirname,directory,subspace);
         load(filename, 'results');
         saccRank(:,j)=results(:,1);
         ensSize=ensSize+1;
         catch
         end 
      end
      for j=1:ensSize
        ensTops=sum((saccRank(:,[1:j])==1),2);
        corr2Score(z,1,j,k)=sum(ensTops.*(j-ensTops))/49999/j/j;
        corr2Score(z,2,j,k)=sum(min(ensTops,j-ensTops))/(j-ceil(j/2))/49999;
        avgAcc=sum(ensTops)/49999/j;
        corr2Score(z,3,j,k)=avgAcc;
        corr2Score(z,4,j,k)=1 - ((sum(ensTops.*(j-ensTops))/j)/ (49999*(j-1)*(avgAcc)*(1-avgAcc)));
        corr2Score(z,5,j,k)=var(ensTops/j);
        corr2Score(z,6,j,k)=1-var(ensTops/j)/(avgAcc*(1-avgAcc));
        corr2Score(z,7,j,k)=1-mean((j-ensTops).*(j-1-ensTops)/j/(j-1))/mean((j-ensTops)/j);
      end
   end
end
