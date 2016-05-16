function [ classResult ] = classScore( blobMeasure, expDim )
%classScore computes teh score of an image classification by overlapping
%areas
%   Detailed explanation goes here

    blankI = zeros(1080,1920);
    blobMap = blankI;
    for b = 1:size(blobMeasure,1)
        xmin = blobMeasure(b).BoundingBox(1);
        ymin = blobMeasure(b).BoundingBox(2);
        xmax = blobMeasure(b).BoundingBox(3)+xmin;
        ymax = blobMeasure(b).BoundingBox(4)+ymin;
        blobMap(ymin:ymax, xmin:xmax) = 1;
        
    end
    expMap = blankI;
    for t = 1:size(expDim,2)
       xminE = str2double(expDim{t}{1});
       xmaxE = str2double(expDim{t}{3});
       yminE = str2double(expDim{t}{2});
       ymaxE = str2double(expDim{t}{4});
       
       expMap(yminE:ymaxE, xminE:xmaxE) = 1;
        
        
    end
     
    classResult =1- sum(sum(abs(blobMap - expMap)))/(2*sum(sum(expMap)));

        
    


end

