% Create some variables
image_folder = 'D:\Tsuki-No-Mi\Extra\ALL';
filenames = dir(fullfile(image_folder, '*.jpg'));
total_images = numel(filenames);
pre_pca_input_nodes = 6534;

allCroppedGrayResizedUnrolled = zeros(total_images, pre_pca_input_nodes);
disease = zeros(total_images, 1);

for i=1:total_images %loop through 5196 images
    full_name = fullfile(image_folder, filenames(i).name);
%     Img = imread(full_name);
%     ImgGray = rgb2gray(Img);
    ImgGray = imread(full_name);
    m = size(ImgGray, 1);
    n = size(ImgGray, 2);
    croppedGray = imcrop(ImgGray, [idivide(uint16(n),2,'ceil')-60, idivide(uint16(m),2,'ceil')-75, 119, 149]);
    croppedGrayResized = imresize(croppedGray, [99 66]);
    allCroppedGrayResizedUnrolled(i,:) = croppedGrayResized(:);
    if filenames(i).name(3) == ']'
        disease(i) = str2double(filenames(i).name(2));
    else
        disease(i) = str2double(strcat(filenames(i).name(2), filenames(i).name(3)));
    end
end

save imageToMat.mat allCroppedGrayResizedUnrolled disease;
% subplot(1,2,1)
% imshow(ImgGray)
% title('Original Image')
% subplot(1,2,2)
% imshow(croppedGray)
% title('Cropped Image')
% Save the variable x,y,z into one *.mat file
%save toMat.mat Imggray