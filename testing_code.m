
bestBBS = bbs(1:200,:);
windowSize = [160 96];
for i = 1: length(bestBBS)
    
    y1 = bestBBS(i,1);
    y2 = bestBBS(i,1)+bestBBS(i,3)-1;
    x1 = bestBBS(i,2);
    x2 = bestBBS(i,2)+bestBBS(i,4)-1;
    
    window11 = double(I(x1:x2,y1:y2));
    
    for winScale = 1:winScale/2:0.125
        for winX = x1:windowSize(2)/2:x2
            for winY = y1: windowSize(1)/2:y2
                
                
                
                % Left off here for sliding window withing objectr proposal
                
                
            end
        end
    end
    
    
    


    
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