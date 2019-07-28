

alexSacc_cMat=sparse(confusionmat(soln,results(:,2)));
alexSacc_cMat=alexSacc_cMat(2:1001,2:1001)


alexSacc_truePred=diag(alexSacc_cMat)
alexSacc_badIndex=find(alexSacc_truePred<5)

alexEns_truePred=diag(alexEns_cMat)
alexEns_badIndex=find(alexEns_truePred<5)

alexEns_improvIdx=find(alexEns_improv>4)
alexEns_worseIdx=find(alexEns_worse>4)




for i=[alexEns_improvIdx']
net.meta.classes.description(find(alexBase_cMat(i,:)>1))'
alexBase_cMat(i,find(alexBase_cMat(i,:)>1))
end
for i=[alexEns_improvIdx']
net.meta.classes.description(find(alexSacc_cMat(i,:)>1))'
alexSacc_cMat(i,find(alexSacc_cMat(i,:)>1))
end
for i=[alexEns_improvIdx']
net.meta.classes.description(find(alexEns_cMat(i,:)>1))'
alexEns_cMat(i,find(alexEns_cMat(i,:)>1))
end






load('/home/jsl18/myMatlab/matconvnet-1.0-beta24/googlenet2/base/result.mat')
r152Base_cMat=sparse(confusionmat(soln,results(:,2)));
r152Base_cMat=r152Base_cMat(2:1001,2:1001)
load('/home/jsl18/myMatlab/matconvnet-1.0-beta24/googlenet2/run_7/result_390.mat')
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


r152Base_cMat=sparse(confusionmat(soln,results(:,2)));
r152Base_cMat=r152Base_cMat(2:1001,2:1001)
load('/home/jsl18/myMatlab/matconvnet-1.0-beta24/resnet152/run_7/result_390.mat')
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

for i=[r152Sacc_badIndex']
net.meta.classes.description(find(r152Base_cMat(i,:)>1))'
r152Base_cMat(i,find(r152Base_cMat(i,:)>1))
end
for i=[r152Sacc_badIndex']
net.meta.classes.description(find(r152Sacc_cMat(i,:)>1))'
r152Sacc_cMat(i,find(r152Sacc_cMat(i,:)>1))
end
for i=[r152Sacc_badIndex']
net.meta.classes.description(find(r152Ens_cMat(i,:)>1))'
r152Ens_cMat(i,find(r152Ens_cMat(i,:)>1))
end

r152Ens_diff=r152Ens_truePred-r152Base_truePred
r152Ens_imprvIdx=find(r152Ens_diff>4)
r152Ens_worseIdx=find(r152Ens_diff<-4)


net.meta.classes.description(r152Ens_worseIdx)'
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


