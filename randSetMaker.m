cd('D:\tbgra\Documents\MATLAB\Image Processing\finalProject\images2')

pics = dir('*.bmp');

personSet = struct;% struct to hold image data
for n=1:length(pics)
    eval([ 'personSet.P' num2str(n) '=((imread(pics(n).name)));']);
    
end