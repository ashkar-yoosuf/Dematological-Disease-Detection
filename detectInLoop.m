%% Initialization
clear ; close all; clc
%LOADING TRAINED NN PARAMETERS
load('nn_params_imagesToMat.mat');
load('mu_sigma_imageToMat.mat');
%load('mu_sigma_5196.mat');
load('U_S_imageToMat.mat');
%load('U_S_5196.mat');

title = 'Detected Disease';
[iconInfo,iconcmap] = imread('info.png');

input_layer_size  = 1000;
hidden_layer1_size = 140;
num_labels = 10;

Theta1 = reshape(nn_params(1:hidden_layer1_size * (input_layer_size + 1)), ...
                 hidden_layer1_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer1_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer1_size + 1));
% Creating the webcam object.
cam = webcam('USB2.0 Camera');
% Capturing one frame to get its size.
videoFrame = snapshot(cam);
frameSize = size(videoFrame);

for i = 1:20
    detect(frameSize, cam, Theta1, Theta2, mu, sigma, U, title, iconInfo, iconcmap)
end


% Clean up.
clear cam;