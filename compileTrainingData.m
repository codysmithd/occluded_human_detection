clear all
close all


%% Compile dataset
tic

cd('D:\tbgra\Documents\MATLAB\training_data\INRIAPerson\INRIAPerson\96X160H96\Train\pos')

pics = dir('*.png');

gmmTraining = struct;% struct to hold image data
for n=1:length(pics)
    eval([ 'testData.P' num2str(n) '=(rgb2gray(imread(pics(n).name)));']);
    groups(n) = 1;
end

cd('D:\tbgra\Documents\MATLAB\training_data\pedestrians128x64')

pics = dir('*.ppm');

for g=1:length(pics)
    eval([ 'testData.P' num2str(g+n) '=(rgb2gray(imread(pics(g).name)));']);
    eval([ 'testData.P' num2str(g+n) '= bilinearInterpolation(testData.P' num2str(g+n) ',[160 96]);']); 
    groups(n+g) = 1;
end



cd('D:\tbgra\Documents\MATLAB\training_data\INRIAPerson\INRIAPerson\Train\neg')

pics = dir();

for k=3:length(pics)
    eval([ 'testData.P' num2str(g+n+k-3) '=(rgb2gray(imread(pics(k).name)));']);
    eval([ 'testData.P' num2str(g+n+k-3) '= bilinearInterpolation(testData.P' num2str(g+n+k-3) ',[160 96]);']); 
    groups(n+g+k-3) = 0;
end
toc

%% Features 
tic
cd ('D:\tbgra\Documents\MATLAB\Image Processing\finalProject')

for i = 1:(length(fieldnames(testData)))
    
    eval([ 'image = testData.P' num2str(i) ';']);
    hog = extractHOGFeatures(image,'CellSize',[10 10], 'NumBins',9,'BlockSize',[5 5] );
    lbp = extractLBPFeatures(image, 'CellSize',[24 24]);
    features(i,:) = [hog lbp];
end

toc


% load('features.mat');
% load('trainingSet.mat');

%% PCA reduction
tic
pcaOpts.zero_mean       = 'true';
pcaOpts.unit_variance   = 'true';
pcaOpts.p               = 0.99;
[pcaFeatures,pcaParams] = reduction_pca(features,pcaOpts);

toc

save('trainingData.mat','features','pcaFeatures','pcaParams','testData','groups');
