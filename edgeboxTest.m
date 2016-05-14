% Edge Box test

clear all
close all

load('personSet');

opts=edgesTrain();                % default options (good settings)
opts.modelDir='models/';          % model will be in models/forest
opts.modelFnm='modelBsds';        % model name
opts.nPos=5e2; opts.nNeg=5e2;     % decrease to speedup training
opts.useParfor=1;                 % parallelize if sufficient memory
opts.split = 'entropy';
opts.normRad = 2;

%% train edge detector (~20m/8Gb per tree, proportional to nPos/nNeg)
tic, model=edgesTrain(opts); toc; % will load model if already trained

%% set detection parameters (can set after training)
model.opts.multiscale=1;          % for top accuracy set multiscale=1
model.opts.sharpen=2;             % for top speed set sharpen=0
model.opts.nTreesEval=6;          % for top speed set nTreesEval=1
model.opts.nThreads=6;            % max number threads for evaluation
model.opts.nms=1;                 % set to true to enable nms
model.opts.stride = 4;

%% 

opts = edgeBoxes();
opts.alpha = .5;     % step size of sliding window search
opts.beta  = .8;     % nms threshold for object proposals
opts.minScore = .05;  % min score of boxes to detect
opts.maxBoxes = 1e4;  % max number of boxes to detect

I = imread('person.png');
G = imread('person_013.bmp');
H = imread('person_014.bmp');
F = imread('person_015.bmp');



tic, bbs=edgeBoxes(I,model,opts); toc


E1 = edgesDetect(I,model);




%% Plots
figure

for i=1:50
    im(1-E1)
    hold on
    rectangle('Position', bbs(i,1:4),'Curvature',0.2,'EdgeColor','b',...
    'LineWidth',1)
    pause;
    hold off
end



