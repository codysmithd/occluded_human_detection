clear all
close all

load('svmModel.mat');
load('nbModel.mat');

mkdir('../results/svm');
mkdir('../results/nb');
mkdir('../results/svm_heat');
mkdir('../results/nb_heat');

base_path = '../imageSets/INRIAPerson/Test/';
test_image_path = strcat(base_path, 'pos/');
test_image_meta_path = strcat(base_path, 'annotations/');
C = 1;

pics = dir(strcat(test_image_path, '*.png'));
svm_classResult = zeros(size(pics));
nb_classResult = zeros(size(pics));
for pic = pics'
    
    % Get image and human box annotations
    
    image = imread(strcat(test_image_path, pic.name));
    
    [~, filename, ~] = fileparts(pic.name);
    raw_data = fileread(strcat(test_image_meta_path, filename, '.txt'));
    tokens = regexp(raw_data, '\((?<xmin>\d*), (?<ymin>\d*)\) - \((?<xmax>\d*), (?<ymax>\d*)\)', 'tokens');
    
    
    
    
    % Get blob, fuse with image and write to results
    [humanBlob_svm, humanMap_svm,svmBlobs] = getHumanBlob(image, svmStruct, model);
    [ svm_classResult(C,:) ] = classScore( svmBlobs, tokens );
    
    [humanBlob_nb, humanMap_nb, nbBlobs] = getHumanBlob(image, nbStruct, model);
    [ nb_classResult(C,:) ] = classScore( nbBlobs, tokens );
    
    imwrite(imfuse(image,humanBlob_svm,'falsecolor','Scaling','independent','ColorChannels',[1 2 0]), strcat('../results/svm/',pic.name));
    imwrite(imfuse(image,humanBlob_nb,'falsecolor','Scaling','independent','ColorChannels',[1 2 0]), strcat('../results/nb/',pic.name));
    imwrite(imfuse(image,humanMap_svm,'falsecolor','Scaling','independent','ColorChannels',[1 2 0]), strcat('../results/svm_heat/',pic.name));
    imwrite(imfuse(image,humanMap_nb,'falsecolor','Scaling','independent','ColorChannels',[1 2 0]), strcat('../results/nb_heat/',pic.name));
    % Check our blob vs the actual TODO
    C = C+1;
end

save('humanClassResults','svm_classResult','nb_classResult');