function [ n ] = householderNormal(u,v)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
u=u/norm(u);
v=v/norm(v);
d=size(u,1);
n=zeros(d,1);
uTv=u'*v;
angle=sqrt((1-uTv)/2);
for i=1:d
   n(i)=(u(i)-v(i))/2/angle;
end
end

