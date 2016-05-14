%% learning GMM
clear all
close all

load('trainingData.mat');


%% Train SVM
groups2 = ones(size(pcaFeatures,1),1);
svmStruct = fitcsvm(pcaFeatures,groups2','KernelScale','auto','Standardize',true,'OutlierFraction',0.5);

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

%% set detection parameters (can set after training)
model.opts.multiscale=1;          % for top accuracy set multiscale=1
model.opts.sharpen=2;             % for top speed set sharpen=0
model.opts.nTreesEval=6;          % for top speed set nTreesEval=1
model.opts.nThreads=6;            % max number threads for evaluation
model.opts.nms=1;                 % set to true to enable nms
model.opts.stride = 4;

%% 

opts = edgeBoxes();
opts.alpha = .65;     % step size of sliding window search
opts.beta  = .9;     % nms threshold for object proposals
opts.eta = .9996;
opts.minScore = .01;  % min score of boxes to detect
opts.maxBoxes = 1e4;  % max number of boxes to detect

G = imread('person.png');
H = imread('person_013.bmp');
F = imread('person_014.bmp');
I = imread('person_015.bmp');

bbs=edgeBoxes(I,model,opts); 

%% window testing 

bestBBS = bbs(1:200,:);

for i = 1: length(bestBBS)

    y1 = bestBBS(i,1);
    y2 = bestBBS(i,1)+bestBBS(i,3)-1;
    x1 = bestBBS(i,2);
    x2 = bestBBS(i,2)+bestBBS(i,4)-1;

    window11 = double(I(x1:x2,y1:y2));
    window11Re = bilinearInterpolation(window11, [160 96]);

    hog = extractHOGFeatures(window11Re,'CellSize',[10 10], 'NumBins',9,'BlockSize',[10 10] );
    lbp = extractLBPFeatures(window11Re, 'CellSize',[24 24]);
    tempFeatures = [hog,lbp];
    
    % PCA
    zmData = tempFeatures -(ones(size(tempFeatures))*diag(pcaParams.M));
    vecData = zmData ./ (ones(size(zmData))*diag(pcaParams.S));
    pcaTempData = vecData * pcaParams.W;

    [predictResult(i,1), predictResult(i,2), predictResult(i,3)] = predict(svmStruct,pcaTempData);

end
I = double(rgb2gray(I));
%% Overlap

dummyI = zeros(size(I));
heatMap = zeros(size(I));

for i=1:length(predictResult)
    
    if (predictResult(i,2)>= 0)
        y1 = bestBBS(i,1);
        y2 = bestBBS(i,1)+bestBBS(i,3)-1;
        x1 = bestBBS(i,2);
        x2 = bestBBS(i,2)+bestBBS(i,4)-1;

        windowPan = ones(bestBBS(i,4),bestBBS(i,3))*((1/abs(predictResult(i,2))));
        
        heatMap(x1:x2,y1:y2) = heatMap(x1:x2,y1:y2) + windowPan;
    end
    
    
    
end

heatMap = heatMap/max(max(heatMap));
th1 = mean(mean(heatMap));
th2 = 0.5;

binHeat1 = im2bw(heatMap,th1);
binHeat2 = im2bw(heatMap,th2);



%% plotting

labeledImage = bwlabel(binHeat2, 8);
blobMeasurements = regionprops(labeledImage, binHeat2, 'all');
numberOfBlobs = size(blobMeasurements, 1);
boundaries = bwboundaries(binHeat2);
boundaries = boundaries{1};

figure
imshow(I,[])
hold on
plot(boundaries(:,2),boundaries(:,1),'g','LineWidth',2)


        
   

