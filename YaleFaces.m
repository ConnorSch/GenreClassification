%% SVD Exploration Images
% initialization
clear all, close all, clc

% Cropped Yale Faces
myMainDir = 'CroppedYale';
myFolds = dir(fullfile(myMainDir));

iter = 1;
for j = 4:length(myFolds)
    baseFoldname = myFolds(j).name;
    fullFoldname = fullfile(myMainDir,baseFoldname);
    myDir = fullfile(myMainDir,baseFoldname);
    myFiles = dir(fullfile(fullFoldname));
    for i = 1:length(myFiles)
        basefilename = myFiles(i).name;
        fullFileName = fullfile(myDir,basefilename);
        if isfile(fullFileName)
            im = imread(fullFileName);
            CF(:,iter) = im(:);
            iter = iter + 1;
        end
    end
end

[U,S,V] = svd(double(CF));

%% Playing with the SVD
figure()
sig = diag(S);
[M,N] = size(CF);

title('Singular Values and PCA Modes for Ideal Case')
subplot(3,2,1), plot(sig(1:100),'ko','Linewidth',[1.5])
%axis([0 7 0 2*10^4])

subplot(3,2,2), semilogy(sig(1:100),'ko','Linewidth',[1.5])
%axis([0 7 0 2*10^4])

xtest = linspace(1,M,M);
subplot(3,1,2) 
plot(xtest,U(:,1),'k',xtest,U(:,2),'k--',xtest,U(:,3),'k:','Linewidth',[2]) 
legend('mode 1','mode 2','mode 3','Location','NorthWest') 

subplot(3,1,3)
t = linspace(1,N,N);
plot(t, V(:,1),'k',t, V(:,2),'k--',t, V(:,3),'k:','Linewidth',[2])
legend('mode 1','mode 2','mode 3','Location','NorthWest') 

%% Recreate an image
[r,c] = size(im);
rank = 10;
LRA = U(:,1:rank)*S(1:rank,1:rank) * V(:,1:rank)'; % Low Rank Approximation of images
image = reshape(LRA(:,1),r,c);

for j = 1:9
   subplot(3,3,j)
   ef = reshape(U(:,j),r,c);
   pcolor(ef),axis off, shading interp, colormap(hot)
end

%% Uncropped Images
MainDir = 'UnCroppedyalefaces';
Files = dir(fullfile(MainDir));
iter2 = 1;
for k = 1:length(Files)
    basefile = Files(i).name;
    fullFile = fullfile(MainDir,basefile);
    if isfile(fullFile)
        im2 = imread(fullFile);
        UCF(:,iter2) = im2(:);
        iter2 = iter2 + 1;
    end
end

[U2,S2,V2] = svd(double(UCF));

%% Recreate an image
[r2,c2] = size(im2);
rank = 1;
LRA2 = U2(:,1:rank)*S2(1:rank,1:rank) * V2(:,1:rank)'; % Low Rank Approximation of images
image2 = reshape(LRA2(:,1),r2,c2);
%imshow(uint8(image2))

figure()
sig2 = diag(S2);
[M,N] = size(UCF);

title('Singular Values and PCA Modes for Ideal Case')
subplot(3,2,1), plot(sig2(1:9),'ko','Linewidth',[1.5])
%axis([0 7 0 2*10^4])

subplot(3,2,2), semilogy(sig2(1:9),'ko','Linewidth',[1.5])
%axis([0 7 0 2*10^4])

figure()
for j = 1:3
   subplot(1,3,j)
   ef = reshape(U2(:,j),r2,c2);
   pcolor(ef),axis off, shading interp, colormap(hot)
end
