function[badFlipProb, paramlist, counter] = badFlip(k,d,theta,runs,o,density,vec1,vec2)
% Arguments are k (projected dimensionality), d (data dimensionality),
% theta (angular separation of true and sample means), runs (number of
% random projections to try), and o (orthonormalsed (1) or Gaussian (0),
% or binary (-1) or sparse (-2)).
% If 2 or 3 arguments are input then the defaults for runs and theta are
% 10000 and pi/4 respectively, and Gaussian R is used.

if(nargin == 2)
    theta = pi/4; % For random mu comment out this line.
    runs = 10000;
    o = 0;
    density=1;
    vec1=[1; zeros(d-1,1)];
    vec2=[1; 0; zeros(d-2,1)];
elseif(nargin == 3)
    runs = 10000; o = 0; density=1;
    vec1=[1; zeros(d-1,1)];
    vec2=[1; 0; zeros(d-2,1)];
elseif(nargin < 2)
    disp('Too few arguments input - please type help badFlip')
    exit
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mu = [1 zeros(1,d-1)];
muhat = [cos(theta) sin(theta) zeros(1,d-2)];
dotprod = mu*muhat';
%rot=eye(d);
%rot = randproj(d,d);
if (density == 0 ) 
    rot = eye(d); 
else    
    rot = sparseproj(d,density);
end
if (nargin<8)
    mu = (rot*mu')';
    muhat = (rot*muhat')';
else
    mu=vec1';
    muhat=(cos(theta)*vec1+sin(theta)*vec2)';
end
u1=mu;
u2=muhat;
angle = acos(dotprod/(norm(mu)*norm(muhat)));

%specially tailored rotation.
%v=sqrt([1/2; 1/(2*(d-1))*randSign(d-1)]);
%temp=randSign(1);
%v=[sqrt((d-1)/(2*d))*temp; sqrt((d+1)/(2*(d^2-d)))*temp; sqrt((d+1)/(2*(d^2-d)))*randSign(d-2)];
%v=[1/sqrt(2) randSign(d-1)'/sqrt(2*(d-1))]';
%mu=mu-2*(mu*v)*v';
%muhat=muhat-2*(muhat*v)*v';


%angle=theta;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

paramlist=zeros(2,runs);
counter = zeros(1,runs);

    for i=1:runs
        if o==2
            R = randproj_n(k,d); % orthonormalised
        %    H= eye(d,d);
        elseif o==1
            R = randproj(k,d); % orthonormalised
        %    H= eye(d,d);
        elseif o==0
            R = randproj_(k,d); % Gaussian
        %    H= eye(d,d);
        elseif o==-1       
            R = randproj_b(k,d);  % try one of Achlioptas' matrices
        %    H= eye(d,d);
        elseif o == -2	    
            R = randproj_3(k,d); % try Achlioptas' sparse matrix
        %    H= eye(d,d);
        elseif o==-3
            R = randsubspace(k,d);
%            v= householder_sv(d);
%            mu=mu-2*(mu*v)*v';
%            muhat=muhat-2*(muhat*v)*v';
        elseif o==-4
            R = unrandsubspace(k,d,density);
        elseif o==-5
            mu=u1;
            muhat=u2;
            R = randsubspace(k,d);
            v= householder_sv(d);
            n = householderNormal(mu',v);
            mu=mu-2*(mu*n)*n';
            muhat=muhat-2*(muhat*n)*n';
        elseif o==-6
            mu=u1;
            muhat=u2;
            R = randsubspace(k,d);
            for j=1:2
                v= householder_sv(d);
                mu=mu-2*(mu*v)*v';
                muhat=muhat-2*(muhat*v)*v';
            end
        elseif o==-7
            mu=u1;
            muhat=u2;
            R = randsubspace(k,d);
            for j=1:k
                v= householder_sv(d);
                mu=mu-2*(mu*v)*v';
                muhat=muhat-2*(muhat*v)*v';
            end
     
            
        else disp('Error: unknown RP matrix');
        end          
        Rmu = (R*mu')';
        Rmuhat = (R*muhat')';
        Rdotprod = Rmu*Rmuhat';
        if(Rdotprod/dotprod < 0)
            counter(1,i) = 1;
        end
        if(Rdotprod == 0)
            counter(1,i) = 0.5;
        end

        Rangle = acos(Rdotprod/(norm(Rmu)*norm(Rmuhat)));
        paramlist(1,i) = angle;
        paramlist(2,i) = Rangle;
    end

badflips = sum(counter(1,:));
if(badflips > 0)
    angles = counter.*paramlist(1,:);
    Rangles = counter.*paramlist(2,:);
end
badFlipProb = badflips/runs;
%paramlist





