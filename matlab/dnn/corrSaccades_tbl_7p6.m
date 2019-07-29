diversityMeas.m
dirnames={'alexnet', 'googlenet', 'resnet50','resnet152'}

subspaces=[450 430 410 390 370 350 330 310 290 270 250 200 150];
directories=1:24;
kSize=size(subspaces,2);
dSize=size(directories,2);
corrScore=zeros(4,6,dSize*(dSize-1)/2,kSize);
for z=1:4
   dirname=dirnames{z}

   for k=1:kSize
       subspace=subspaces(k);
       runCount=0;
       for j=1:dSize
           directory1=directories(j);
           filename=sprintf('%s/run_%d/result_%d.mat',dirname,directory1,subspace);
           load(filename, 'results');
           sacc1Rank=results(:,1);
           sacc1TopIdx=(sacc1Rank==1);
           sacc1NotIdx=(sacc1Rank>1);
           for i=j+1:dSize;
            runCount=runCount+1;    
            directory2=directories(i);
            filename=sprintf('%s/run_%d/result_%d.mat',dirname,directory2,subspace);
            load(filename, 'results');
            sacc2Rank=results(:,1);
         try
         N11=sum(sacc2Rank(sacc1TopIdx)==1);
         N00=sum(sacc2Rank(sacc1NotIdx)>1);
         N01=sum(sacc2Rank(sacc1NotIdx)==1);
         N10=sum(sacc2Rank(sacc1TopIdx)>1);
         corrScore(z,1,runCount,k)=(N11*N00 - N01*N10)/(sqrt((N11+N10)*(N01+N00)*(N11+N01)*(N10+N00)));
         corrScore(z,2,runCount,k)=(2*N00) / (N10+N01+2*N00); 
         corrScore(z,3,runCount,k)=(N00*N11 - N01*N10)/(N00*N11 + N10*N01); 
         corrScore(z,4,runCount,k)=(N01+N10)/(N00 + N11 + N10 + N01); 
         corrScore(z,5,runCount,k)=N00/(N00 + N11 + N10 + N01);
         corrScore(z,6,runCount,k)=2*(N00*N11 - N01*N10)/((N11+N10)*(N01+N00) + (N11+N01)*(N10+N00)); 
         catch
         end 
      end
   end
   mean(reshape(corrScore(z,:,:,k),6,dSize*(dSize-1)/2),2)
   end
end
