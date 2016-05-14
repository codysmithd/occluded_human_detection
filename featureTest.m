% Testing HOG, SURF, LBP

clear all
close all


load('trainingSet.mat')

% load('gmmTraining.mat');



%% Features 

for i = 1:(length(fieldnames(testData)))
    
    eval([ 'image = testData.P' num2str(i) ';']);
    hog = extractHOGFeatures(image,'CellSize',[4 4], 'NumBins',4,'BlockSize',[2 2] );
    lbp = extractLBPFeatures(image, 'CellSize',[32 32]);
    features(i,:) = [hog,lbp];
end