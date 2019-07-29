%% Class Analyser helper script used to help generate tables 7.9 - 7.11 and D.8 - D.16
% Run this after processResult_confMat, and change the folders accordingly,
% because this was done once off, i didn't bother to automate it

load('base/result.mat')
r152Base_cMat=sparse(confusionmat(soln,results(:,2)));
r152Base_cMat=r152Base_cMat(2:1001,2:1001)
load('run_1/result_390.mat')
r152Sacc_cMat=sparse(confusionmat(soln,results(:,2)));
r152Sacc_cMat=r152Sacc_cMat(2:1001,2:1001)
r152Ens_cMat=sparse(confusionmat(soln,topSolution(:,10,4)));

r152Base_truePred=diag(r152Base_cMat);
r152Base_badIndex=find(r152Base_truePred<6)

r152Sacc_truePred=diag(r152Sacc_cMat);
r152Sacc_badIndex=find(r152Sacc_truePred<6)

r152Ens_truePred=diag(r152Ens_cMat);
r152Ens_badIndex=find(r152Ens_truePred<6)


net.meta.classes.description(r152Ens_badIndex)'

for i=[r152Ens_badIndex']
net.meta.classes.description(find(r152Base_cMat(i,:)>1))'
r152Base_cMat(i,find(r152Base_cMat(i,:)>1))
end
for i=[r152Ens_badIndex']
net.meta.classes.description(find(r152Sacc_cMat(i,:)>1))'
r152Sacc_cMat(i,find(r152Sacc_cMat(i,:)>1))
end
for i=[r152Ens_badIndex']
net.meta.classes.description(find(r152Ens_cMat(i,:)>1))'
r152Ens_cMat(i,find(r152Ens_cMat(i,:)>1))
end

r152Ens_diff=r152Ens_truePred-r152Base_truePred
r152Ens_imprvIdx=find(r152Ens_diff>4)
r152Ens_worseIdx=find(r152Ens_diff<-4)

for i=[r152Ens_worseIdx']
net.meta.classes.description(find(r152Base_cMat(i,:)>1))'
r152Base_cMat(i,find(r152Base_cMat(i,:)>1))
end
for i=[r152Ens_worseIdx']
net.meta.classes.description(find(r152Sacc_cMat(i,:)>1))'
r152Sacc_cMat(i,find(r152Sacc_cMat(i,:)>1))
end
for i=[r152Ens_worseIdx']
net.meta.classes.description(find(r152Ens_cMat(i,:)>1))'
r152Ens_cMat(i,find(r152Ens_cMat(i,:)>1))
end

net.meta.classes.description(r152Ens_imprvIdx)'

for i=[r152Ens_imprvIdx']
net.meta.classes.description(find(r152Base_cMat(i,:)>1))'
r152Base_cMat(i,find(r152Base_cMat(i,:)>1))
end
for i=[r152Ens_imprvIdx']
net.meta.classes.description(find(r152Sacc_cMat(i,:)>1))'
r152Sacc_cMat(i,find(r152Sacc_cMat(i,:)>1))
end
for i=[r152Ens_imprvIdx']
net.meta.classes.description(find(r152Ens_cMat(i,:)>1))'
r152Ens_cMat(i,find(r152Ens_cMat(i,:)>1))
end

