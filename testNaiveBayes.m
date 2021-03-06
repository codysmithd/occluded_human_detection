clear all
close all

load('nbModel.mat');

dir_path = '../imageSets/INRIAPerson/Test/pos/';

image = imread(strcat(dir_path,'crop_000006.png'));
tic
[humanBlob, humanMap, blobMeasure] = getHumanBlob(image, nbStruct, model);
toc
%% Output image with human blob

figure
imshow(imfuse(image,humanBlob,'falsecolor','Scaling','independent','ColorChannels',[1 2 0]));
hold on

figure
imshow(imfuse(image,humanMap,'falsecolor','Scaling','independent','ColorChannels',[1 2 0]));
hold on