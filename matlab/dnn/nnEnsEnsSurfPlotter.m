ensSize=18;
[X,Y]=meshgrid(1:ensSize,subspaces);

surf(X,Y,meanL(:,:,2)')
hold on
surf(X,Y,baseRes152Top1*ones(noSubspace,ensSize),'AlphaData',0.7*ones(noSubspace,ensSize),'FaceAlpha','flat','FaceColor','red')
surf(X,Y,baseRes152Top1*ones(noSubspace,ensSize),'AlphaData',0.8*ones(noSubspace,ensSize),'FaceAlpha','flat','FaceColor','red')
zlim([0.35 0.8])
xlim([1 15])
ylim([150 450])
hold on
surf(X,Y,baseRes152Top1*ones(noSubspace,ensSize),'AlphaData',0.8*ones(noSubspace,ensSize),'FaceAlpha','flat','FaceColor','red')
%baseRes152Top1=36225/49999
%baseRes152Top3=(36225+5301+1997)/49999
%baseRes152Top5=(36225+5301+1997+1066+738)/49999

subplot(1,3,1);
[X,Y]=meshgrid(ensL,subspaces)
surf(X,Y,meanL(:,:,2)')
hold on
surf(X,Y,baseRes152Top1*ones(noSubspace,ensSize),'AlphaData',0.8*ones(noSubspace,ensSize),'FaceAlpha','flat','FaceColor','red')
surf(X,Y,baseRes152Top1*ones(noSubspace,ensSize),'AlphaData',0.7*ones(noSubspace,ensSize),'FaceAlpha','flat','FaceColor','red')
zlim([0.35 0.75])
xlim([1 15])
ylim([150 450])
zlim([0.4 0.8])
title('Top 1','FontSize',10)
xlabel({'Ensemble';'Size'},'FontSize',9)
ylabel({'Projection';'Dimension'},'FontSize',9)
zlabel('Accuracy','FontSize',9)


subplot(1,3,2);
surf(X,Y,meanL(:,:,3)')
hold on
surf(X,Y,baseRes152Top3*ones(noSubspace,ensSize),'AlphaData',0.7*ones(noSubspace,ensSize),'FaceAlpha','flat','FaceColor','red')
surf(X,Y,baseRes152Top3*ones(noSubspace,ensSize),'AlphaData',0.8*ones(noSubspace,ensSize),'FaceAlpha','flat','FaceColor','red')
zlim([0.5 0.9])
xlim([1 15])
ylim([150 450])
title('Top 3','FontSize',10)
xlabel({'Ensemble';'Size'},'FontSize',9)
ylabel({'Projection';'Dimension'},'FontSize',9)
zlabel('Accuracy','FontSize',9)


subplot(1,3,3);
surf(X,Y,meanL(:,:,4)')
hold on
surf(X,Y,baseRes152Top5*ones(noSubspace,ensSize),'AlphaData',0.8*ones(noSubspace,ensSize),'FaceAlpha','flat','FaceColor','red')
surf(X,Y,baseRes152Top5*ones(noSubspace,ensSize),'AlphaData',0.7*ones(noSubspace,ensSize),'FaceAlpha','flat','FaceColor','red')
zlim([0.55 0.95])
xlim([1 15])
ylim([150 450])
title('Top 5','FontSize',10)
xlabel({'Ensemble';'Size'},'FontSize',9)
ylabel({'Projection';'Dimension'},'FontSize',9)
zlabel('Accuracy','FontSize',9)