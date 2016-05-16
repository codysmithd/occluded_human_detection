clear all
close all

load('nbModel.mat');
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

dir_path = '../imageSets/INRIAPerson/Test/pos/';
I = imread(strcat(dir_path,'crop_000006.png'));

bbs=edgeBoxes(I,model,opts); 

%% window testing 
tic

bestBBS = bbs(1:25,:);
windowW = 96;
windowH = 160;
count = 1;

widthI = size(I,2);
heightI = size(I,1);

predictResult = zeros(length(bestBBS),1);

for i = 1: length(bestBBS)
    
    y1 = bestBBS(i,1);
    y2 = bestBBS(i,1)+bestBBS(i,3)-1;
    x1 = bestBBS(i,2);
    x2 = bestBBS(i,2)+bestBBS(i,4)-1;
    
    if(((x2-x1)*(y2-y1))<(widthI*heightI*0.25))
    
    
        window11 = double(I(x1:x2,y1:y2));


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
                lbpV = [lbpV lbp];
            end
            hogC = [hogC, hogV];
            lbpC = [lbpC,lbpV];
        end

        tempFeatures = [lbpC,hogC];
        [~, predictResult(i,1:2), ~] = predict(nbStruct, tempFeatures);
    end
end
toc

%% Overlap
I = double(rgb2gray(I));

dummyI = zeros(size(I));
heatMap = zeros(size(I));

for i=1:length(predictResult)
    
    if (predictResult(i,1) == 1)
        x1 = bestBBS(i,1);
        x2 = bestBBS(i,1)+bestBBS(i,3)-1;
        y1 = bestBBS(i,2);
        y2 = bestBBS(i,2)+bestBBS(i,4)-1;

        windowPan = ones((y2-y1),x2-x1)*((1/abs((predictResult(i)))));
%         if (((y2-1)<size(heatMap,1)) && ((x2-1)<size(heatMap,2)))
            
            heatMap(y1:y2-1,x1:x2-1) = heatMap(y1:y2-1,x1:x2-1) + windowPan;
            
%         else
            
%         end
    end
    
    
    
end
figure
im(heatMap)

heatMap = heatMap/max(max(heatMap));
th1 = mean(max(heatMap));
th2 = max(mean(heatMap));

binHeat1 = im2bw(heatMap,th1);
binHeat2 = im2bw(heatMap,th2);

figure
subplot(1,2,1)
im(binHeat1)
subplot(1,2,2)
im(binHeat2)

%% plotting

labeledImage = bwlabel(binHeat1, 8);
blobMeasurements = regionprops(labeledImage, binHeat1, 'all');
numberOfBlobs = size(blobMeasurements, 1);
boundaries = bwboundaries(binHeat1);
boundaries = boundaries{1};

figure
imshow(I,[])
hold on
plot(boundaries(:,2),boundaries(:,1),'g','LineWidth',2)
