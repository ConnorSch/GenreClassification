%% Music Genre Classification
% initialization
clear all, close all, clc

mainDir = 'MusicClassification/ArtistClassification';

myFiles = dir(fullfile(mainDir));
num_seg = 10;
iter = 1;
for i = 4:length(myFiles)
    fullFileName = fullfile(mainDir,myFiles(i).name);
    if isfile(fullFileName)
        info = audioinfo(fullFileName);   
        mid = round(info.TotalSamples/2);
        step = 5*info.SampleRate;
        points = linspace(-num_seg/2,num_seg/2,num_seg+1);
        for j = 1:num_seg
            point = mid+points(j)*step;
            sample = double([point,point + step]);
            audio = audioread(fullFileName,sample);
            if size(audio,2) > 1
                audio = sum(audio,2)/size(audio,2);
            end
            AC(:,iter) = audio;
            iter = iter + 1; 
        end
    end
end

a_size = num_seg*5;

%% Pull Out Train and Test sets
q1 = randperm(a_size);
q2 = randperm(a_size);
q3 = randperm(a_size);

samplerate = 0.8;
tests = samplerate*a_size;
AIC = AC(:,1:a_size);
AIC_train = AIC(:,q1(1:tests));
AIC_test = AIC(:,q1(tests+1:end));

PJ = AC(:,a_size+1:2*a_size);
PJ_train = PJ(:,q2(1:tests));
PJ_test = PJ(:,q2(tests+1:end));

SG = AC(:,2*a_size+1:end);
SG_train = SG(:,q3(1:tests));
SG_test = SG(:,q3(tests+1:end));

%% Create spectograms with Matlab
AIC_sp = [];
PJ_sp = [];
SG_sp = [];
for i = 1:a_size*samplerate
   aics = spectrogram(AIC_train(:,i));
   pjs = spectrogram(PJ_train(:,i));
   sgs = spectrogram(SG_train(:,i));
   
   AIC_s(:,i) = abs(aics(:));
   PJ_s(:,i) = abs(pjs(:));
   SG_s(:,i) = abs(sgs(:));  
end

for j = 1:size(AIC_test,2)
   aicst = spectrogram(AIC_test(:,j));
   pjst = spectrogram(PJ_test(:,j));
   sgst = spectrogram(SG_test(:,j));
   
   AIC_st(:,j) = abs(aicst(:));
   PJ_st(:,j) = abs(pjst(:));
   SG_st(:,j) = abs(sgst(:));  
end

X = [AIC_s, PJ_s, SG_s];
Xtest = [AIC_st,PJ_st,SG_st];

%% Singular Values and Principal Components

[U,S,V] = svd(X,'econ');

figure()
sig = diag(S);
[M,N] = size(X);

title('Singular Values')
subplot(1,2,1), plot(sig(1:a_size),'ko','Linewidth',[1.5])
%axis([0 7 0 2*10^4])

subplot(1,2,2), semilogy(sig(1:a_size),'ko','Linewidth',[1.5])
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
  plot(1:a_size*samplerate,V(1:a_size*samplerate,j),'ko-') 
  V2 = [2,5,8];
  subplot(3,3,V2(j)) 
  plot(a_size*samplerate+1:2*a_size*samplerate,V(a_size*samplerate+1:2*a_size*samplerate,j),'ko-')
  V3 = [3,6,9];
  subplot(3,3,V3(j))
  plot(2*a_size*samplerate+1:3*a_size*samplerate,V(2*a_size*samplerate+1:3*a_size*samplerate,j),'ko-')
end
subplot(3,3,1), title('Alice in Chains') 
subplot(3,3,2), title('Pearl Jam')
subplot(3,3,3), title('Soundgarden')

%% Create training and test sets

    
numFeat = 40;

samplesize = samplerate * a_size;
xtrain = V(:,1:numFeat);
xtest = U'*Xtest;

ctrain = [repmat({'Alice In Chains'},[samplesize,1]);repmat({'Pearl Jam'},[samplesize,1]);repmat({'Soundgarden'},[samplesize,1])];
truth = [repmat({'Alice In Chains'},[a_size-samplesize,1]);repmat({'Pearl Jam'},[a_size-samplesize,1]);repmat({'Soundgarden'},[a_size-samplesize,1])];

svm.mod = fitcecoc(V,ctrain);
pre = predict(svm.mod,xtest');

num_correct = 0;
for k = 1:length(truth)
   if strcmp(pre{k},truth{k})
        num_correct = num_correct + 1;
   end
end
percentage = (num_correct/length(truth))*100
