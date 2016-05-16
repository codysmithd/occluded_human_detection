function [ humanBlob, humanMap, blobMeasurements ] = getHumanBlob( image, classifier, model )
%GETRESULT Summary of this function goes here
%   Detailed explanation goes here

%% set detection parameters (can set after training)
model.opts.multiscale=1;          % for top accuracy set multiscale=1
model.opts.sharpen=2;             % for top speed set sharpen=0
model.opts.nTreesEval=6;          % for top speed set nTreesEval=1
model.opts.nThreads=6;            % max number threads for evaluation
model.opts.nms=1;                 % set to true to enable nms
model.opts.stride = 4;

%% set options

opts = edgeBoxes();
opts.alpha = .65;     % step size of sliding window search
opts.beta  = .9;     % nms threshold for object proposals
opts.eta = .9996;
opts.minScore = .01;  % min score of boxes to detect
opts.maxBoxes = 1e4;  % max number of boxes to detect

bbs = edgeBoxes(image, model, opts); 

%% window testing 
tic

bestBBS = bbs(1:25,:);

widthI = size(image,2);
heightI = size(image,1);

predictResult = zeros(length(bestBBS),1);

for i = 1: length(bestBBS)
    
    y1 = bestBBS(i,1);
    y2 = bestBBS(i,1)+bestBBS(i,3)-1;
    x1 = bestBBS(i,2);
    x2 = bestBBS(i,2)+bestBBS(i,4)-1;
    
    if(((x2-x1)*(y2-y1))<(widthI*heightI))
    
    
        window11 = double(image(x1:x2,y1:y2));


        window11Re = bilinearInterpolation(window11, [160 96] );

        hogC = [];
        lbpC = [];
        tempSize = size(window11Re);
        stepY = round(tempSize(1)/2);
        stepX = round(tempSize(2)/2);
        for c = 1:stepX:tempSize(2)
            hogV = [];
            lbpV = [];
            for v = 1:stepY:tempSize(1)
                hog = extractHOGFeatures(window11Re,'CellSize',[6 6], 'NumBins',4,'BlockSize',[3 3] );
                lbp = extractLBPFeatures(window11Re, 'CellSize',[15 15]);

                hogV = [hogV, hog];
                lbpV = [lbpV, lbp];
            end
            hogC = [hogC, hogV];
            lbpC = [lbpC,lbpV];
        end

        tempFeatures = [lbpC,hogC];
        [~, predictResult(i,1:2),~] = predict(classifier,tempFeatures);
    end

end
toc

I = double(rgb2gray(image));

%% Overlap

dummyI = zeros(size(I));
heatMap = zeros(size(I));

for i=1:length(predictResult)
    
    if (predictResult(i)~= 0)
        x1 = bestBBS(i,1);
        x2 = bestBBS(i,1)+bestBBS(i,3)-1;
        y1 = bestBBS(i,2);
        y2 = bestBBS(i,2)+bestBBS(i,4)-1;

        windowPan = ones((y2-y1),x2-x1)*((1/abs((predictResult(i)))));
        heatMap(y1:y2-1,x1:x2-1) = heatMap(y1:y2-1,x1:x2-1) + windowPan;
    end
    
end

heatMap = (heatMap/max(max(heatMap)));

th4 = multithresh(heatMap,6);

binHeat4 = imquantize(heatMap,th4);
th5 = (max(max(binHeat4))-(max(max(binHeat4))-mean(mean(binHeat4)))*.25)/max(max(binHeat4));
binHeat5 = im2bw(binHeat4/max(max(binHeat4)),th5);

SE = strel('octagon',15);
humanMap = heatMap;
humanBlob = imdilate(binHeat5,SE);


labeledImage = bwlabel(humanBlob, 8); 
blobMeasurements = regionprops(labeledImage, I, 'all');


end

