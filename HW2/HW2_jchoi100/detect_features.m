function [ rows,cols ] = detect_features( image )
    %%%
    % Computer Vision 600.461/661 Assignment 2
    % Args:
    %   image (ndarray): The input image to detect features on. Note: this is NOT the image name or image path.
    % Returns:
    %   rows: A list of row indices of detected feature locations in the image
    %   cols: A list of col indices of detected feature locations in the image
    %%%
    
    image = double(rgb2gray(image));
    dx = [-1 0 1; -1 0 1; -1 0 1];
    dy = dx';
    sigma = 3;
    radius = 5;
    T = 1100;
    k = 0.04;
    
    Ix = conv2(image, dx, 'same');
    Iy = conv2(image, dy, 'same');    

    % Gaussian filter with size 6 * sigma and of min size 1 x 1
    g = fspecial('gaussian', max(1, fix(6 * sigma)), sigma);
    
    Ix2 = conv2(Ix.^2, g, 'same');
    Iy2 = conv2(Iy.^2, g, 'same');
    Ixy = conv2(Ix.*Iy, g, 'same');
    
    % CS = (Ix2.*Iy2 - Ixy.^2)./(Ix2 + Iy2); % Harris corner measure
    CS = (Ix2.*Iy2 - 0.5  * Ixy.^2) - k * (Ix2 + Iy2);
    [rows, cols] = nonmaxsuppts(CS, radius, T, image);
end