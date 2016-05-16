clear all
close all

load('svmModel.mat');
load('nbModel.mat');

mkdir('../results/svm');
mkdir('../results/nb');

base_path = '../imageSets/INRIAPerson/Test/';
test_image_path = strcat(base_path, 'pos/');
test_image_meta_path = strcat(base_path, 'annotations/');

pics = dir(strcat(test_image_path, '*.png'));

for pic = pics'
    
    % Get image and human box annotations
    
    image = imread(strcat(test_image_path, pic.name));
    
    [~, filename, ~] = fileparts(pic.name);
    raw_data = fileread(strcat(test_image_meta_path, filename, '.txt'));
    tokens = regexp(raw_data, '\((?<xmin>\d*), (?<ymin>\d*)\) - \((?<xmax>\d*), (?<ymax>\d*)\)', 'tokens');
    
    xmin = str2double(tokens{1}{1});
    ymin = str2double(tokens{1}{2});
    xmax = str2double(tokens{1}{3});
    ymax = str2double(tokens{1}{4});
    
    % Get blob, fuse with image and write to results
    humanBlob_svm = getHumanBlob(image, svmStruct, model);
    humanBlob_nb = getHumanBlob(image, nbStruct, model);
    imwrite(imfuse(image,humanBlob_svm,'falsecolor','Scaling','independent','ColorChannels',[1 2 0]), strcat('../results/svm/',pic.name));
    imwrite(imfuse(image,humanBlob_nb,'falsecolor','Scaling','independent','ColorChannels',[1 2 0]), strcat('../results/nb/',pic.name));
    
    % Check our blob vs the actual TODO
    
end