%% Helper script to plot the surface graphs figure 7.5

[X2,Y2]=meshgrid(1:ensSize,subspaces);
[X,Y]=meshgrid(1:ensSize,subspaces);
[X1,Y1]=meshgrid(1:ensSize,subspaces);
ensSize=15
top1Alex=reshape(recordsAlexnet(1,1,1:ensSize,:,:),ensSize,noSubspace)/49999
top3Alex=reshape(sum(recordsAlexnet(1,1:3,1:ensSize,:,:)),ensSize,noSubspace)/49999
top5Alex=reshape(sum(recordsAlexnet(1,1:5,1:ensSize,:,:)),ensSize,noSubspace)/49999
top5Google=reshape(sum(recordsGooglenet(1,1:5,1:ensSize,:,:)),ensSize,noSubspace)/49999
top3Google=reshape(sum(recordsGooglenet(1,1:3,1:ensSize,:,:)),ensSize,noSubspace)/49999
top1Google=reshape(recordsGooglenet(1,1,1:ensSize,:,:),ensSize,noSubspace)/49999
top1Res=reshape(recordsResnet50(1,1,1:ensSize,:,:),ensSize,noSubspace)/49999
top3Res=reshape(sum(recordsResnet50(1,1:3,1:ensSize,:,:)),ensSize,noSubspace)/49999
top5Res=reshape(sum(recordsResnet50(1,1:5,1:ensSize,:,:)),ensSize,noSubspace)/49999
top1Res152=reshape(recordsResnet152(1,1,1:ensSize,:,:),ensSize,noSubspace)/49999
top3Res152=reshape(sum(recordsResnet152(1,1:3,1:ensSize,:,:)),ensSize,noSubspace)/49999
top5Res152=reshape(sum(recordsResnet152(1,1:5,1:ensSize,:,:)),ensSize,noSubspace)/49999

%% Values obtained from base simulation 
% TODO: rather than hardcode the values, process the base simulation and
% calculate them. 
baseAlexTop1=27348/49999
baseGoogleTop1=32731/49999
baseResTop1=35192/49999
baseResTop3=(35192+5423+2160)/49999
baseResTop5=(35192+5423+2160+1213+841)/49999
baseAlexTop3=(27348+5833+2658)/49999
baseAlexTop5=(27348+5833+2658+1707+1234)/49999
baseGoogleTop5=(32731+5883+2494+1422+933)/49999
baseGoogleTop3=(32731+5883+2494)/49999
baseRes152Top1=36225/49999
baseRes152Top3=(36225+5301+1997)/49999
baseRes152Top5=(36225+5301+1997+1066+738)/49999

%% Surface plotting loop, 
% We plot the reference plane twice to darken the plane while keeping it translucent. 
% This is a stop gap approach that works well enough.

figure
subplot(4,3,1)
surf(X2,Y2,top1Alex')
hold on
surf(X2,Y2,baseAlexTop1*ones(noSubspace,ensSize),'AlphaData',0.8*ones(noSubspace,ensSize),'FaceAlpha','flat','FaceColor','red')
surf(X2,Y2,baseAlexTop1*ones(noSubspace,ensSize),'AlphaData',0.7*ones(noSubspace,ensSize),'FaceAlpha','flat','FaceColor','red')
zlim([0.35 0.75])
xlim([1 15])
ylim([150 450])

subplot(4,3,4)
surf(X,Y,top1Google')
hold on
surf(X,Y,baseGoogleTop1*ones(noSubspace,ensSize),'AlphaData',0.7*ones(noSubspace,ensSize),'FaceAlpha','flat','FaceColor','red')
surf(X,Y,baseGoogleTop1*ones(noSubspace,ensSize),'AlphaData',0.8*ones(noSubspace,ensSize),'FaceAlpha','flat','FaceColor','red')

zlim([0.35 0.75])
xlim([1 15])
ylim([150 450])

subplot(4,3,7)
surf(X,Y,top1Res')
hold on
surf(X,Y,baseResTop1*ones(noSubspace,ensSize),'AlphaData',0.7*ones(noSubspace,ensSize),'FaceAlpha','flat','FaceColor','red')
surf(X,Y,baseResTop1*ones(noSubspace,ensSize),'AlphaData',0.8*ones(noSubspace,ensSize),'FaceAlpha','flat','FaceColor','red')
zlim([0.35 0.75])
xlim([1 15])
ylim([150 450])

subplot(4,3,10)
surf(X1,Y1,top1Res152')
hold on
surf(X1,Y1,baseRes152Top1*ones(noSubspace,ensSize),'AlphaData',0.7*ones(noSubspace,ensSize),'FaceAlpha','flat','FaceColor','red')
surf(X1,Y1,baseRes152Top1*ones(noSubspace,ensSize),'AlphaData',0.8*ones(noSubspace,ensSize),'FaceAlpha','flat','FaceColor','red')
zlim([0.35 0.75])
xlim([1 15])
ylim([150 450])

subplot(4,3,1)
title('Top 1, Alexnet','FontSize',10)
xlabel({'Ensemble';'Size'},'FontSize',9)
ylabel({'Projection';'Dimension'},'FontSize',9)
zlabel('Accuracy','FontSize',9)

subplot(4,3,4)
title('Top 1, Googlenet','FontSize',10)
xlabel({'Ensemble';'Size'},'FontSize',9)
ylabel({'Projection';'Dimension'},'FontSize',9)
zlabel('Accuracy','FontSize',9)

subplot(4,3,7)
title('Top 1, Resnet-50','FontSize',10)
xlabel({'Ensemble';'Size'},'FontSize',9)
ylabel({'Projection';'Dimension'},'FontSize',9)
zlabel('Accuracy','FontSize',9)

subplot(4,3,10)
title('Top 1, Resnet-152','FontSize',10)
xlabel({'Ensemble';'Size'},'FontSize',9)
ylabel({'Projection';'Dimension'},'FontSize',9)
zlabel('Accuracy','FontSize',9)

%figure
subplot(4,3,2)
surf(X2,Y2,top3Alex')
hold on
surf(X2,Y2,baseAlexTop3*ones(noSubspace,ensSize),'AlphaData',0.8*ones(noSubspace,ensSize),'FaceAlpha','flat','FaceColor','red')
surf(X2,Y2,baseAlexTop3*ones(noSubspace,ensSize),'AlphaData',0.7*ones(noSubspace,ensSize),'FaceAlpha','flat','FaceColor','red')
zlim([0.5 0.9])
xlim([1 15])
ylim([150 450])

subplot(4,3,5)
surf(X,Y,top3Google')
hold on
surf(X,Y,baseGoogleTop3*ones(noSubspace,ensSize),'AlphaData',0.7*ones(noSubspace,ensSize),'FaceAlpha','flat','FaceColor','red')
surf(X,Y,baseGoogleTop3*ones(noSubspace,ensSize),'AlphaData',0.8*ones(noSubspace,ensSize),'FaceAlpha','flat','FaceColor','red')

zlim([0.5 0.9])
xlim([1 15])
ylim([150 450])

subplot(4,3,8)
surf(X,Y,top3Res')
hold on
surf(X,Y,baseResTop3*ones(noSubspace,ensSize),'AlphaData',0.7*ones(noSubspace,ensSize),'FaceAlpha','flat','FaceColor','red')
surf(X,Y,baseResTop3*ones(noSubspace,ensSize),'AlphaData',0.8*ones(noSubspace,ensSize),'FaceAlpha','flat','FaceColor','red')
zlim([0.5 0.9])
xlim([1 15])
ylim([150 450])

subplot(4,3,11)
surf(X1,Y1,top3Res152')
hold on
surf(X1,Y1,baseRes152Top3*ones(noSubspace,ensSize),'AlphaData',0.7*ones(noSubspace,ensSize),'FaceAlpha','flat','FaceColor','red')
surf(X1,Y1,baseRes152Top3*ones(noSubspace,ensSize),'AlphaData',0.8*ones(noSubspace,ensSize),'FaceAlpha','flat','FaceColor','red')
zlim([0.5 0.9])
xlim([1 15])
ylim([150 450])
subplot(4,3,2)
title('Top 3, Alexnet','FontSize',10)
xlabel({'Ensemble';'Size'},'FontSize',9)
ylabel({'Projection';'Dimension'},'FontSize',9)
zlabel('Accuracy','FontSize',9)

subplot(4,3,5)
title('Top 3, Googlenet','FontSize',10)
xlabel({'Ensemble';'Size'},'FontSize',9)
ylabel({'Projection';'Dimension'},'FontSize',9)
zlabel('Accuracy','FontSize',9)

subplot(4,3,8)
title('Top 3, Resnet-50','FontSize',10)
xlabel({'Ensemble';'Size'},'FontSize',9)
ylabel({'Projection';'Dimension'},'FontSize',9)
zlabel('Accuracy','FontSize',9)

subplot(4,3,11)
title('Top 3, Resnet-152','FontSize',10)
xlabel({'Ensemble';'Size'},'FontSize',9)
ylabel({'Projection';'Dimension'},'FontSize',9)
zlabel('Accuracy','FontSize',9)


%figure
subplot(4,3,3)
surf(X2,Y2,top5Alex')
hold on
surf(X2,Y2,baseAlexTop5*ones(noSubspace,ensSize),'AlphaData',0.8*ones(noSubspace,ensSize),'FaceAlpha','flat','FaceColor','red')
surf(X2,Y2,baseAlexTop5*ones(noSubspace,ensSize),'AlphaData',0.7*ones(noSubspace,ensSize),'FaceAlpha','flat','FaceColor','red')
zlim([0.55 0.95])
xlim([1 15])
ylim([150 450])

subplot(4,3,6)
surf(X,Y,top5Google')
hold on
surf(X,Y,baseGoogleTop5*ones(noSubspace,ensSize),'AlphaData',0.7*ones(noSubspace,ensSize),'FaceAlpha','flat','FaceColor','red')
surf(X,Y,baseGoogleTop5*ones(noSubspace,ensSize),'AlphaData',0.8*ones(noSubspace,ensSize),'FaceAlpha','flat','FaceColor','red')

zlim([0.55 0.95])
xlim([1 15])
ylim([150 450])

subplot(4,3,9)
surf(X,Y,top5Res')
hold on
surf(X,Y,baseResTop5*ones(noSubspace,ensSize),'AlphaData',0.7*ones(noSubspace,ensSize),'FaceAlpha','flat','FaceColor','red')
surf(X,Y,baseResTop5*ones(noSubspace,ensSize),'AlphaData',0.8*ones(noSubspace,ensSize),'FaceAlpha','flat','FaceColor','red')
zlim([0.55 0.95])
xlim([1 15])
ylim([150 450])


subplot(4,3,12)
surf(X1,Y1,top5Res152')
hold on
surf(X1,Y1,baseRes152Top5*ones(noSubspace,ensSize),'AlphaData',0.7*ones(noSubspace,ensSize),'FaceAlpha','flat','FaceColor','red')
surf(X1,Y1,baseRes152Top5*ones(noSubspace,ensSize),'AlphaData',0.8*ones(noSubspace,ensSize),'FaceAlpha','flat','FaceColor','red')
zlim([0.55 0.95])
xlim([1 15])
ylim([150 450])

subplot(4,3,3)
title('Top 5, Alexnet','FontSize',10)
xlabel({'Ensemble';'Size'},'FontSize',9)
ylabel({'Projection';'Dimension'},'FontSize',9)
zlabel('Accuracy','FontSize',9)

subplot(4,3,6)
title('Top 5, Googlenet','FontSize',10)
xlabel({'Ensemble';'Size'},'FontSize',9)
ylabel({'Projection';'Dimension'},'FontSize',9)
zlabel('Accuracy','FontSize',9)

subplot(4,3,9)
title('Top 5, Resnet-50','FontSize',10)
xlabel({'Ensemble';'Size'},'FontSize',9)
ylabel({'Projection';'Dimension'},'FontSize',9)
zlabel('Accuracy','FontSize',9)

subplot(4,3,12)
title('Top 5, Resnet-152','FontSize',10)
xlabel({'Ensemble';'Size'},'FontSize',9)
ylabel({'Projection';'Dimension'},'FontSize',9)
zlabel('Accuracy','FontSize',9)
