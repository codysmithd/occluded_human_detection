%Image dataset creation
clear all
close all

cd('D:\tbgra\Documents\MATLAB\training_data\INRIAPerson\INRIAPerson\96X160H96\Train\pos')

pics = dir('*.png');

gmmTraining = struct;% struct to hold image data
for n=1:length(pics)
    eval([ 'testData.P' num2str(n) '=(rgb2gray(imread(pics(n).name)));']);
    groups{n} = 'Human';
end

cd('D:\tbgra\Documents\MATLAB\training_data\pedestrians128x64')

pics = dir('*.ppm');

for g=1:length(pics)
    eval([ 'testData.P' num2str(g+n) '=(rgb2gray(imread(pics(g).name)));']);
    eval([ 'testData.P' num2str(g+n) '= bilinearInterpolation(testData.P' num2str(g+n) ',[160 96]);']); 
    groups{n+g} = 'Human';
end



cd('D:\tbgra\Documents\MATLAB\training_data\INRIAPerson\INRIAPerson\Train\neg')

pics = dir();

for k=3:length(pics)
    eval([ 'testData.P' num2str(g+n+k-3) '=(rgb2gray(imread(pics(k).name)));']);
    eval([ 'testData.P' num2str(g+n+k-3) '= bilinearInterpolation(testData.P' num2str(g+n+k-3) ',[160 96]);']); 
    groups{n+g+k-3} = 'nonHuman';
end
