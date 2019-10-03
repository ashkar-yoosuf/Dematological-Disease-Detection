%% Initialization
clear ; close all; clc

% Creating the webcam object.
cam = webcam('USB2.0 Camera');
% Capturing one frame to get its size.
videoFrame = snapshot(cam);
frameSize = size(videoFrame);

    % Creating the video player object.
    videoPlayer = vision.VideoPlayer('Position', [100 100 [frameSize(2), frameSize(1)]+30]);
    %Detection and Tracking
    runLoop = true;
    i = 0;
    j = 0;
    % mkdir('Detected', int2str(j));
    aCGRUInDetection = zeros(5, 6534);
    while runLoop
        % Getting the next frame, capturing face and saving in folder
        videoFrame = snapshot(cam);
        im = im2bw(videoFrame);
        im_unrolled = im(:);
        baseFileName = sprintf('[1](%d).jpg', j);
        folder = 'D:\Tsuki-No-Mi\Extra\[10]Warts';
        lowerBound = 307200 - 260000;
        upperBound = 307200 - 220000;
        if lowerBound < size(im_unrolled, 1) - sum(im_unrolled) && size(im_unrolled, 1) - sum(im_unrolled) < upperBound && i <= 535
            i = i + 1;
            if i > 36
                j = j + 1;
                imInvert =~im;
                s = regionprops(imInvert, 'boundingbox');
                X = reshape(struct2array(s), 4,[]);
                [M, I] = max(sum(X(3:end, :)), [], 2);
                videoFrameGray = rgb2gray(videoFrame);
                croppedGray = imcrop(videoFrameGray, struct2array(s(I)));

                m = size(croppedGray, 1);
                n = size(croppedGray, 2);
                %Crop again to 150X120, resize to 99X66 and save in
                %croppedGrayResized
%                 croppedGray = imcrop(croppedGray, [idivide(uint16(n),2,'ceil')-60, idivide(uint16(m),2,'ceil')-75, 119, 149]);
%                 croppedGrayResized = imresize(croppedGray, [99 66]);
%                 aCGRUInDetection(j,:) = croppedGrayResized(:);  

                %just checking the saved image
                fullFileName = fullfile(folder, baseFileName);
                imwrite(croppedGray, fullFileName);
            end
        elseif ~(lowerBound < size(im_unrolled, 1) - sum(im_unrolled) && size(im_unrolled, 1) - sum(im_unrolled) < upperBound)...
                && i > 535
            runLoop = false;
        end
        % Displaying the video frame using the video player object.
        step(videoPlayer, videoFrame);
        %Checking whether the video player window has been closed.
        %runLoop = isOpen(videoPlayer);
    end
    release(videoPlayer)