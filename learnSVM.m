%% learning GMM
clear all
close all

load('trainingData.mat');

%% Train SVM
tic
groups2 = ones(size(features,1),1);
svmStruct = fitcsvm(features,groups2','KernelScale','auto','Standardize',true,'OutlierFraction',0.15);
toc
%% Object Proposal

opts=edgesTrain();                % default options (good settings)
opts.modelDir='models/';          % model will be in models/forest
opts.modelFnm='modelBsds';        % model name
opts.nPos=5e2; opts.nNeg=5e2;     % decrease to speedup training
opts.useParfor=1;                 % parallelize if sufficient memory
opts.split = 'entropy';
opts.normRad = 2;

%% train edge detector (~20m/8Gb per tree, proportional to nPos/nNeg)
tic, model=edgesTrain(opts); toc; % will load model if already trained
%cd ('D:\tbgra\Documents\MATLAB\Image Processing\finalProject')

save('svmModel.mat','model','svmStruct','-v7.3');