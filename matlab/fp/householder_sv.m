function[vectorN] = householder_sv(d)
%vectorN=zeros(d,1);
%vectorN=[sqrt((1+(1/d))/2)*randSign(1); randSign(d-1)*sqrt(1/(2*d))];
vectorN=[randSign(d)*sqrt(1/d)];
%vectorN=[ones(d,1)*-sqrt(1/d)];
%transformMatrix=eye(d)-2*vectorN*vectorN';