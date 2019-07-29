%% Helper script to results into a sparse matrix for processing (used with python)

dirnames={'alexnet', 'googlenet', 'resnet50', 'resnet152'}

subspaces=[450 430 410 390 370 350 330 310 290 270 250 200 150];
directories=1:26;
kSize=size(subspaces,2);
dSize=size(directories,2);
for z=1:4
   dirname=dirnames{z}
   for j=1:dSize
      directory=directories(j)
      for k=1:kSize
         try
         subspace=subspaces(k);
         	
         filename=sprintf('%s/run_%d/result_%d.mat',dirname,directory,subspace)
         
         best=sparse(50000,1000);
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
   		
         filename=sprintf('%s/run_%d/sparse_%d.mat',dirname,directory,subspace)
         save(filename,'best');
         catch
         end 
      end
   end
end
