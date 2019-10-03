function detect(frameSize, cam, Theta1, Theta2, mu, sigma, U, title, iconInfo, iconcmap)
    % Creating the video player object.
    videoPlayer = vision.VideoPlayer('Position', [100 100 [frameSize(2), frameSize(1)]+30]);
    %Detection and Tracking
    runLoop = true;
    i = 0;
    j = 0;
    % mkdir('Detected', int2str(j));
    aCGRUInDetection = zeros(30, 6534);
    while runLoop
        % Getting the next frame, capturing face and saving in folder
        videoFrame = snapshot(cam);
        im = im2bw(videoFrame);
        im_unrolled = im(:);
        baseFileName = sprintf('[1](%d).jpg', j);
        folder = 'D:\Tsuki-No-Mi\Detected';
        lowerBound = 307200 - 260000;
        upperBound = 307200 - 220000;
        if lowerBound < size(im_unrolled, 1) - sum(im_unrolled) && size(im_unrolled, 1) - sum(im_unrolled) < upperBound && i <= 40
            i = i + 1;
            if i > 11
                j = j + 1;
                imInvert =~im;
                s = regionprops(imInvert, 'boundingbox');
                X = reshape(struct2array(s), 4,[]);
                [I] = maxProdBySum(X); %Gives the Index where max Prod by sum of 3rd and 4th elements exist
                %[M, I] = max(sum(X(3:end, :)), [], 2);
                videoFrameGray = rgb2gray(videoFrame);
                croppedGray = imcrop(videoFrameGray, struct2array(s(I)));

                m = size(croppedGray, 1);
                n = size(croppedGray, 2);
                %Crop again to 150X120, resize to 99X66 and save in
                %croppedGrayResized
                croppedGray = imcrop(croppedGray, [idivide(uint16(n),2,'ceil')-60, idivide(uint16(m),2,'ceil')-75, 119, 149]);
                croppedGrayResized = imresize(croppedGray, [99 66]);
                aCGRUInDetection(j,:) = croppedGrayResized(:);  

                %just checking the saved image
                fullFileName = fullfile(folder, baseFileName);
                imwrite(croppedGray, fullFileName);
            end
        elseif ~(lowerBound < size(im_unrolled, 1) - sum(im_unrolled) && size(im_unrolled, 1) - sum(im_unrolled) < upperBound)...
                && i > 40
            runLoop = false;
        end
        % Displaying the video frame using the video player object.
        step(videoPlayer, videoFrame);
        %Checking whether the video player window has been closed.
        %runLoop = isOpen(videoPlayer);
    end
    release(videoPlayer)
    %Normalizing aCGRUInDetection
    [X_Norm_Prediction] = featureNormalizePrediction(aCGRUInDetection, mu, sigma);
    %Run PCA
	%[U, S] = pca(X_Norm_Prediction);
	%Dimension Reduction of Faces
	K = 1000;
	Z = projectData(X_Norm_Prediction, U, K);
    %predicting the disease
    pred = predict(Theta1, Theta2, Z);
    pred
    if mode(pred) == 10
        fprintf('\npredicted = Warts\n')
        uiwait(msgbox('Warts',title,'custom',iconInfo,iconcmap,'modal'))
    elseif mode(pred) == 1
        fprintf('\npredicted = Acne\n')
        uiwait(msgbox('Acne',title,'custom',iconInfo,iconcmap,'modal'))
    elseif mode(pred) == 2
        fprintf('\npredicted = Acne-Keratosis\n')
        uiwait(msgbox('Acne-Keratosis',title,'custom',iconInfo,iconcmap,'modal'))
    elseif mode(pred) == 3
        fprintf('\npredicted = Basal-Cell-Carcinoma\n')
        uiwait(msgbox('Basal-Cell-Carcinoma',title,'custom',iconInfo,iconcmap,'modal'))
    elseif mode(pred) == 4
        fprintf('\npredicted = Eczema\n')
        uiwait(msgbox('Eczema',title,'custom',iconInfo,iconcmap,'modal'))
    elseif mode(pred) == 5
        fprintf('\npredicted = Herpes-Zoster\n')
        uiwait(msgbox('Herpes-Zoster!',title,'custom',iconInfo,iconcmap,'modal'))
    elseif mode(pred) == 6
        fprintf('\npredicted = Lichen\n')
        uiwait(msgbox('Lichen',title,'custom',iconInfo,iconcmap,'modal'))
    elseif mode(pred) == 7
        fprintf('\npredicted = Nevi\n')
        uiwait(msgbox('Nevi',title,'custom',iconInfo,iconcmap,'modal'))
    elseif mode(pred) == 8
        fprintf('\npredicted = Psoriasis\n')
        uiwait(msgbox('Psoriasis',title,'custom',iconInfo,iconcmap,'modal'))
    else
        fprintf('\npredicted = Tinea\n')
        uiwait(msgbox('Tinea',title,'custom',iconInfo,iconcmap,'modal'))
    end
end