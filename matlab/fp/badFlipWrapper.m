runGaussian=1;
for runType=1:5
if runType==4
    qScales=[1/2 2/3 3/4];
else
    qScales=[1]
end
for qScale=qScales
for uFactor=[1 2 5 10]
%thetaRun=[1:50]*pi/100;
%runs=10000
runs=100;
%oRun=[0 -3 -5];
oRun=[-7 -6 -5 -3 -2 -1 0 1 2]
d=100
%d=1000;
theta=pi/3;
density=0;
kRun=[1 5 10 20 50 100];
k=20;
densityRun=[1:20]*1/40;
thetaRun=[1:100]/100;

subRun=kRun;

sphericalRun=zeros(size(thetaRun,2),size(subRun,2));
orthNormalRun=zeros(size(thetaRun,2),size(subRun,2));
if(runGaussian==1)
gaussianRun=zeros(size(thetaRun,2),size(subRun,2));
end
achioRun=zeros(size(thetaRun,2),size(subRun,2));
sparsAchioRun=zeros(size(thetaRun,2),size(subRun,2));
subspaceRun=zeros(size(thetaRun,2),size(subRun,2));
subspace2Run=zeros(size(thetaRun,2),size(subRun,2));
subspace3Run=zeros(size(thetaRun,2),size(subRun,2));
subspace4Run=zeros(size(thetaRun,2),size(subRun,2));

if runType>1
    R1=[ones(1,d/uFactor,1) zeros(1,d-(d/uFactor))]; % Sparse UT4=utFactor
elseif runType==1
    R1= [1 zeros(1,d-1)];
    rot=sparseproj(d,1/uFactor);
    R1=R1*rot;
end
%R1=4.^-[1:uFactor]; % Dense UT4 = 
%R1=repmat(R1,1,d/uFactor);  
%R1=randn(1,d);
R1=(R1/norm(R1));

if runType==1
    R2=randn(1,d); %Gaussian UT2 ~ 1
    vectorType='Gaussian Vectors';
elseif runType==2
    R2=randn(1,d); %Gaussian UT2 ~ 1
    vectorType='Binary And Gaussian Vector';
elseif runType==3
    R2=[ones(1,d/uFactor/2) -ones(1,d/uFactor/2) zeros(1,d-(d/uFactor))]; %Dense UT2 = uFactor
    vectorType='Coincided Binary Vectors';
elseif runType==4
    R2=[qScale/sqrt(d/uFactor)*ones(1,d/uFactor/2) -qScale/sqrt(d/uFactor)*ones(1,d/uFactor/2) sqrt((1-(qScale^2))/(d-(d/uFactor)))*ones(1,d-(d/uFactor))]; %Dense UT2 = uFactor
    vectorType='Scaled Binary Vectors';
elseif runType==5
    R2=[zeros(1,d/uFactor) ones(1,d-d/uFactor)]; %Dense UT2 = uFactor
    vectorType='Uncoincided Binary Vector';
end
    R2=(R2/norm(R2));
    R2=R2-(R1*R2')*R1;
    R2=(R2/norm(R2))';
    R1=R1';
%R2=[zeros(1,1); ones(d/uFactor,1); zeros(d-(d/uFactor)-1,1)]';
%R2=[ones(d/uFactor/4,1);-ones(d/uFactor/4,1); zeros(d-(d/uFactor/2),1)]'; %Dense UT2 = uFactor
%qScale=1/2

%R2=[0 randnormalized(1,d-1)];
%R2=R2*rot;

U3T = sum(R1.^3.*R2)*d
U4=sum(R1.^4)*d
UT2=sum((R1.*R2).^2)*d

subspaceSansNullSpace=zeros(size(thetaRun,2),size(subRun,2));
debug=zeros(size(thetaRun,2),runs);

for i=1:size(subRun,2)
    k=subRun(i);
    i
    %density=subRun(i)
    for o=oRun
        tic;        
        for j=1:size(thetaRun,2)
        theta=thetaRun(j)*pi;
        flag=0;
        if o<runGaussian
            flipProb=badFlip_(k,d,theta,runs,o,density,R1,R2);
        
            if o==2
                sphericalRun(j,i)=flipProb;
            elseif o==1
                orthNormalRun(j,i)=flipProb;
            elseif o==0
                gaussianRun(j,i)=flipProb;
            elseif o==-2
                sparsAchioRun(j,i)=flipProb;
            elseif o==-1
                achioRun(j,i)=flipProb;
            elseif o==-3
                subspaceRun(j,i)=flipProb;
                nullProb=1;
                d0=floor(density*d);
                d1=d-d0;
                for l=0:k-1
                    nullProb=nullProb*(d1-l)/(d-l);
                end
                subspaceSansNullSpace(j,i)=flipProb-nullProb/2;
            elseif o==-5
                subspace2Run(j,i)=flipProb;
            elseif o==-6
                subspace3Run(j,i)=flipProb;
            elseif o==-7
                subspace4Run(j,i)=flipProb;
            end
        end

        end
        o
        toc
    end
%subplot(3,3,i), plot(thetaRun,gaussianRun(:,i),thetaRun,othnormalRun(:,i),thetaRun,subspaceRun(:,i));
%subplot(4,5,i), plot([1:100]*1/200,gaussianRun(:,i),[1:100]*1/200,othnormalRun(:,i),[1:100]*1/200,subspaceRun(:,i));
% %subplot(4,5,i), plot(thetaRun,subspaceRun(:,i));
% lgdn={};
% subplot(3,2,i);
% hold on;
% for o=oRun
%             if o==2
%                 plot(thetaRun,sphericalRun(:,i),'--o','MarkerSize',2);
%                 lgdn{end+1}='Spherical';
%             elseif o==1
%                 plot(thetaRun,orthNormalRun(:,i),'--o','MarkerSize',2);
%                 lgdn{end+1}='orthNormal';               
%             elseif o==0
%                 plot(thetaRun,gaussianRun(:,i),'--o','MarkerSize',2);
%                 lgdn{end+1}='gaussian';               
%             elseif o==-2
%                 plot(thetaRun,sparsAchioRun(:,i),'--+','MarkerSize',2);
%                 lgdn{end+1}='Achiolptas (1/6)';     
%             elseif o==-1
%                 plot(thetaRun,achioRun(:,i),'--x','MarkerSize',2);
%                 lgdn{end+1}='Achiolptas (1/2)';    
%             elseif o==-3
%                 plot(thetaRun,subspaceRun(:,i),'--*','MarkerSize',2);
%                 lgdn{end+1}='Subspace (as is)';
%             elseif o==-5
%                 plot(thetaRun,subspace2Run(:,i),'--s','MarkerSize',2);
%                 lgdn{end+1}='Subspace+HouseHolder';
%             elseif o==-6
%                 plot(thetaRun,subspace3Run(:,i),'--+','MarkerSize',2);
%                 lgdn{end+1}='Subspace+2 HouseHolder';
%             elseif o==-7
%                 plot(thetaRun,subspace4Run(:,i),'--.','MarkerSize',2);
%                 lgdn{end+1}='Subspace+k HouseHolder';
%             end
% end
% probEst=exp(-(k)./(U4+UT2*tan(thetaRun*pi).^2));
% plot(thetaRun,probEst,'-');
% %plot(thetaRun,probEst/2);
% lgdn{end+1}='ProbBound Bernstein';
%lgdn{end+1}='1/2 Bernstein';
%Z=sum((R1.*R1 + (R1.*R2).*tan(thetaRun*pi)).^2)*d;
%probEst=exp(-(k)./(2*Z));
%plot(thetaRun,probEst);

%lgdn{end+1}='Bernstein no Triangle';
%Z=max((R1.*R1 + (R1.*R2).*tan(thetaRun*pi)))-min((R1.*R1 + (R1.*R2).*tan(thetaRun*pi)))
%probEst=exp((-2*k)./(Z.^2*d));
%plot(thetaRun,probEst);
%lgdn{end+1}='Mc Dirmids Bound';

%probEst=(U4+UT2*tan(thetaRun*pi).^2)./(k+(U4+UT2*tan(thetaRun*pi).^2));
%plot(thetaRun,probEst);
%lgdn{end+1}='Cantelli Bound';

%probEst=1/2 * exp(-k./U4) + 1/2 * exp(-(k)./(U4+2*UT2*tan(thetaRun*pi).^2));
%plot(thetaRun,probEst,'--');
%lgdn{end+1}='Gut-feel sparse bound';


%probEst=1/2* exp(-(k)./((U4+2/d)+abs(2*UT2+4/d)*tan(thetaRun*pi).^2));
%plot(thetaRun,probEst,'-');
%lgdn{end+1}='Gut-feel dense bound';
% strTitle=sprintf('U4=%.1f, UT2=%.1f, subspace=%d',U4, UT2, k);
% title(strTitle);
% if i==6 
%     legend(lgdn);
% end
% ylim([0 1]);
% drawnow;
end
if runType==4
    myTitle=sprintf('Flipping probability for %s, Sparsity d/s = %d, q = %.1f',vectorType,uFactor,qScale);
else
    myTitle=sprintf('Flipping probability for %s, Sparsity d/s = %d',vectorType,uFactor); 
end
close gcf;
[fig myAxes]=createAxes([3 2],'title',myTitle,'hSize',7.5,'vSize',9);
myPlots=gobjects(1,size(oRun,2));
for i=1:size(subRun,2);
    k=subRun(i);
    axes(myAxes(i));
    xlabel('\theta/\pi (rads)');
    title(sprintf('Subspace=%d',k));
    ylabel('Flipping Probability');
    hold on;
    l=0;
    lgdn={};
    for o=oRun;
        l=l+1;
           if o==2
                myPlots(l)=plot(thetaRun,sphericalRun(:,i),'--o','MarkerSize',2);
                lgdn{end+1}='Spherical';
            elseif o==1
                myPlots(l)=plot(thetaRun,orthNormalRun(:,i),'--o','MarkerSize',2);
                lgdn{end+1}='orthNormal';               
            elseif o==0
                myPlots(l)=plot(thetaRun,gaussianRun(:,i),'--o','MarkerSize',2);
                lgdn{end+1}='Gaussian Random Projection';               
            elseif o==-2
                myPlots(l)=plot(thetaRun,sparsAchioRun(:,i),'--+','MarkerSize',2);
                lgdn{end+1}='Achiolptas (1/6)';     
            elseif o==-1
                myPlots(l)=plot(thetaRun,achioRun(:,i),'--x','MarkerSize',2);
                lgdn{end+1}='Achiolptas (1/2)';    
            elseif o==-3
                myPlots(l)=plot(thetaRun,subspaceRun(:,i),'--*','MarkerSize',2);
                lgdn{end+1}='Random Subspace';
            elseif o==-5
                myPlots(l)=plot(thetaRun,subspace2Run(:,i),'--s','MarkerSize',2);
                lgdn{end+1}='Random Subspace + Householder Transformation';
            elseif o==-6
                myPlots(l)=plot(thetaRun,subspace3Run(:,i),'--+','MarkerSize',2);
                lgdn{end+1}='Subspace+2 HouseHolder';
            elseif o==-7
                myPlots(l)=plot(thetaRun,subspace4Run(:,i),'--.','MarkerSize',2);
                lgdn{end+1}='Subspace+k HouseHolder';
           end
    end

probEst=exp(-(k)./(U4+UT2*tan(thetaRun*pi).^2+U3T*tan(thetaRun*pi)));
myPlots(l+1)=plot(thetaRun,probEst,'-');
%plot(thetaRun,probEst/2);
xlim([0 1]);
ylim([0 1]);
lgdn{end+1}='Theorem Probability Bounds';  
end
legend(myPlots,lgdn,'Location',[0.3 0.01 0.4 0.9/9]);






fileName=sprintf('flipProb_run_%d_sparsity_%d_qScale_%.1f.fig',runType,uFactor,qScale);
savefig(fileName);
fileName=sprintf('flipProb_run_%d_sparsity_%d_qScale_%.1f.eps',runType,uFactor,qScale);
saveas(gcf,fileName,'epsc');


end
end
end