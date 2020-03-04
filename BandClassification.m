%% Music Artist Classification
% initialization
clear all, close all, clc

mainDir = 'MusicClassification/BandClassification';


% Try looping through collecting numerous samples of music
%Country
fullFoldname = fullfile(mainDir,'Country');
myFiles = dir(fullfile(fullFoldname));

num_seg = 10;
iter = 1;
for i = 4:length(myFiles)
    basefilename = myFiles(i).name;
    fullFileName = fullfile(fullFoldname,basefilename);
    if isfile(fullFileName)
        info = audioinfo(fullFileName);   
        mid = round(info.TotalSamples/2);
        step = 5*info.SampleRate;
        points = linspace(-num_seg/2,num_seg/2,num_seg+1);
        for j = 1:num_seg
            point = mid+points(j)*step;
            sample = double([point,point + step]);
            audio = audioread(fullFileName,sample);
            CM(:,iter) = audio;
            iter = iter + 1; 
        end
    end
end

% Grunge
fullFoldname = fullfile(mainDir,'Grunge');
myFiles = dir(fullfile(fullFoldname));

iter = 1;
for i = 4:length(myFiles)
    basefilename = myFiles(i).name;
    fullFileName = fullfile(fullFoldname,basefilename);
    if isfile(fullFileName)
        info = audioinfo(fullFileName);   
        mid = round(info.TotalSamples/2);
        step = 5*info.SampleRate;
        points = linspace(-num_seg/2,num_seg/2,num_seg+1);
        for j = 1:num_seg
            point = mid+points(j)*step;
            sample = double([point,point + step]);
            audio = audioread(fullFileName,sample);
            GM(:,iter) = audio;
            iter = iter + 1; 
        end
    end
end

% RAP
fullFoldname = fullfile(mainDir,'RAP');
myFiles = dir(fullfile(fullFoldname));

iter = 1;
for i = 4:length(myFiles)
    basefilename = myFiles(i).name;
    fullFileName = fullfile(fullFoldname,basefilename);
    if isfile(fullFileName)
        info = audioinfo(fullFileName);   
        mid = round(info.TotalSamples/2);
        step = 5*info.SampleRate;
        points = linspace(-num_seg/2,num_seg/2,num_seg+1);
        for j = 1:num_seg
            point = mid+points(j)*step;
            sample = double([point,point + step]);
            audio = audioread(fullFileName,sample);
            RM(:,iter) = audio;
            iter = iter + 1; 
        end
    end
end

g_size = iter-1;

%% Pull Out Train and Test sets
q1 = randperm(g_size);
q2 = randperm(g_size);
q3 = randperm(g_size);

samplerate = 0.8;
tests = samplerate*g_size;
CM_train = CM(:,q1(1:tests));
CM_test = CM(:,q1(tests+1:end));

GM_train = GM(:,q2(1:tests));
GM_test = GM(:,q2(tests+1:end));

RM_train = RM(:,q3(1:tests));
RM_test = RM(:,q3(tests+1:end));

%% Create spectograms with Matlab
CM_sp = [];
GM_sp = [];
RM_sp = [];
for i = 1:g_size*samplerate
   cs = spectrogram(CM_train(:,i));
   gs = spectrogram(GM_train(:,i));
   rs = spectrogram(RM_train(:,i));
   
   CM_s(:,i) = abs(cs(:));
   GM_s(:,i) = abs(gs(:));
   RM_s(:,i) = abs(rs(:));  
end

for j = 1:size(CM_test,2)
   cst = spectrogram(CM_test(:,j));
   gst = spectrogram(GM_test(:,j));
   rst = spectrogram(RM_test(:,j));
   
   CM_st(:,j) = abs(cst(:));
   GM_st(:,j) = abs(gst(:));
   RM_st(:,j) = abs(rst(:));  
end
X = [CM_s, GM_s, RM_s];
Xtest = [CM_st,GM_st,RM_st];

%% Singular Values and Principal Components
[U,S,V] = svd(X,'econ');

figure()
sig = diag(S);
[M,N] = size(X);

title('Singular Values')
subplot(1,2,1), plot(sig(1:g_size),'ko','Linewidth',[1.5])
%axis([0 7 0 2*10^4])

subplot(1,2,2), semilogy(sig(1:g_size),'ko','Linewidth',[1.5])
%axis([0 7 0 2*10^4])

figure()
for j = 1:6
   subplot(2,3,j)
   plot(U(:,j))
end

figure(3)
for j=1:3
  V1 = [1,4,7];
  subplot(3,3,V1(j)) 
  plot(1:g_size*samplerate,V(1:g_size*samplerate,j),'ko-') 
  V2 = [2,5,8];
  subplot(3,3,V2(j)) 
  plot(g_size*samplerate+1:2*g_size*samplerate,V(g_size*samplerate+1:2*g_size*samplerate,j),'ko-')
  V3 = [3,6,9];
  subplot(3,3,V3(j))
  plot(2*g_size*samplerate+1:3*g_size*samplerate,V(2*g_size*samplerate+1:3*g_size*samplerate,j),'ko-')
end
subplot(3,3,1), title('Johnny Cash') 
subplot(3,3,2), title('Alice in Chains')
subplot(3,3,3), title('Eminem')

%% Create training and test sets

test_runs = 20;
percentage = zeros(1,test_runs);
for l = 1:test_runs 
    
    numFeat = 40;

    samplesize = samplerate * g_size;
    xtrain = V(:,1:numFeat);
    xtest = U'*Xtest;

    ctrain = [repmat({'Johnny Cash'},[samplesize,1]);repmat({'Alice In Chains'},[samplesize,1]);repmat({'Eminem'},[samplesize,1])];
    %ctrain = [ones(samplesize,1);2*ones(samplesize,1);3*ones(samplesize,1)];
    truth = [repmat({'Johnny Cash'},[g_size-samplesize,1]);repmat({'Alice In Chains'},[g_size-samplesize,1]);repmat({'Eminem'},[g_size-samplesize,1])];

    %xtrain = [CM_train,GM_train,RM_train];
    %xtest = [CM_test,GM_test,RM_test];
    %pre = classify(xtest,xtrain,ctrain,'linear');  
    
    svm.mod = fitcecoc(V,ctrain);
    pre = predict(svm.mod,xtest');
    
    num_correct = 0;
    for k = 1:length(truth)
       if strcmp(pre{k},truth{k})
            num_correct = num_correct + 1;
       end
    end
    percentage(l) = (num_correct/length(truth))*100;
end
mean(percentage)
