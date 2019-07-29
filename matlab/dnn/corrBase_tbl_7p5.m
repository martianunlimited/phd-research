dirnames={'alexnet', 'googlenet', 'resnet50','resnet152'}

subspaces=[450 430 410 390 370 350 330 310 290 270 250 200 150];
directories=1:26;
kSize=size(subspaces,2);
dSize=size(directories,2);
corrScore=zeros(4,10,dSize,kSize);
for z=1:4
   dirname=dirnames{z}
   baseFile=sprintf('%s/base/result.mat',dirname)
   load(baseFile,'results');
   baseRank=results(:,1);
   for j=1:dSize
      directory=directories(j)
      for k=1:kSize
         try
         subspace=subspaces(k);
         filename=sprintf('%s/run_%d/result_%d.mat',dirname,directory,subspace);
         load(filename, 'results');
         saccRank=results(:,1);
         corrScore(z,1,j,k)=corr(saccRank(baseRank>0).^-1,baseRank(baseRank>0).^-1);
         corrScore(z,2,j,k)=corr(saccRank(baseRank>0),baseRank(baseRank>0));
         N11=sum(saccRank(baseRank==1)==1);
         N00=sum(saccRank(baseRank>1)>1);
         N01=sum(saccRank(baseRank>1)==1);
         N10=sum(saccRank(baseRank==1)>1);
         corrScore(z,3,j,k)=(N11*N00 - N01*N10)/(sqrt((N11+N10)*(N01+N00)*(N11+N01)*(N10+N00)));
         N11=sum(saccRank(baseRank<4)<4);
         N00=sum(saccRank(baseRank>3)>3);
         N01=sum(saccRank(baseRank>3)<4);
         N10=sum(saccRank(baseRank<4)>3);
         corrScore(z,4,j,k)=(N11*N00 - N01*N10)/(sqrt((N11+N10)*(N01+N00)*(N11+N01)*(N10+N00)));
         N11=sum(saccRank(baseRank<6)<6);
         N00=sum(saccRank(baseRank>5)>5);
         N01=sum(saccRank(baseRank>5)<6);
         N10=sum(saccRank(baseRank<6)>5);
         corrScore(z,5,j,k)=(N11*N00 - N01*N10)/(sqrt((N11+N10)*(N01+N00)*(N11+N01)*(N10+N00)));        
         N11=sum(saccRank(baseRank==1)==1);
         N00=sum(saccRank(baseRank>1)>1);
         N01=sum(saccRank(baseRank>1)==1);
         N10=sum(saccRank(baseRank==1)>1);
         corrScore(z,6,j,k)=(2*N00) / (N10+N01+2*N00); 
         corrScore(z,7,j,k)=(N00*N11 - N01*N10)/(N00*N11 + N10*N01); 
         corrScore(z,8,j,k)=(N01+N10)/(N00 + N11 + N10 + N01); 
         corrScore(z,9,j,k)=N00/(N00 + N11 + N10 + N01);
         corrScore(z,10,j,k)=2*(N00*N11 - N01*N10)/((N11+N10)*(N01+N00) + (N11+N01)*(N10+N00)); 
         catch
         end 
      end
   end
   reshape(corrScore(z,:,:,:),10,dSize,kSize);
end
