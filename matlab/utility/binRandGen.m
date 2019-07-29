function [S]=binRandGen(baseProb,dim,obs)
randomness=baseProb/2;
s=min(floor(((rand(obs,1)-0.5)*randomness+baseProb)*dim),dim);
%s=min(floor(abs(randn(obs,1)*randomness+baseProb)*dim),dim);
%s=ones(obs,1)*floor(baseProb*dim);
counts=sum(s);
I=zeros(1,counts);
J=zeros(1,counts);
counter=1;
for i=1:obs
    I(counter:counter+s(i)-1)=randperm(dim,s(i));
    J(counter:counter+s(i)-1)=ones(1,s(i))*i;
    counter=counter+s(i);
end
S=sparse(I,J,1,dim,obs);
end