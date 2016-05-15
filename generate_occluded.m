% Generates the occluded dataset
% Assumes the location of the INRIA dataset is ../imageSets/INRIAPerson

dir_path = '../imageSets/INRIAPerson/96X160H96/Train/';
files = dir(strcat(dir_path, 'pos/*.png'));

% Create dirs we are going to add to
mkdir(strcat(dir_path, 'left_occluded'));
mkdir(strcat(dir_path, 'right_occluded'));
mkdir(strcat(dir_path, 'top_occluded'));
mkdir(strcat(dir_path, 'bottom_occluded'));

for file = files'
    
    image = imread(strcat(dir_path, 'pos/', file.name));
    
    [height, width, depth] = size(image);
    
    %% Left
    mask = uint8(zeros(height, width, 3));
    mask(:,width/2:end,:) = 1;
    masked_image = image.*mask;
    imwrite(masked_image, strcat(dir_path, 'left_occluded/',file.name));
    
    %% Right
    mask = uint8(zeros(height, width, 3));
    mask(:,1:width/2,:) = 1;
    masked_image = image.*mask;
    imwrite(masked_image, strcat(dir_path, 'right_occluded/',file.name));
    
    %% Top
    mask = uint8(zeros(height, width, 3));
    mask(1:height/2,:,:) = 1;
    masked_image = image.*mask;
    imwrite(masked_image, strcat(dir_path, 'top_occluded/',file.name));
    
    %% Bottom
    mask = uint8(zeros(height, width, 3));
    mask(height/2:end,:,:) = 1;
    masked_image = image.*mask;
    imwrite(masked_image, strcat(dir_path, 'bottom_occluded/',file.name));
    
end