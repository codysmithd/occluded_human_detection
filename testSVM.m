clear all
close all

load('svmModel.mat');

dir_path = '../imageSets/INRIAPerson/Test/pos/';

image = imread(strcat(dir_path,'crop_000006.png'));

humanBlob = getHumanBlob(image, svmStruct, model);

%% Output image with human blob

figure
imshow(imfuse(image,humanBlob,'falsecolor','Scaling','independent','ColorChannels',[1 2 0]));
hold on
