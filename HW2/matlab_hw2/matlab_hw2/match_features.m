function [ matches ] = match_features(feature_coords1,feature_coords2,image1,image2)
    %%% 
	% Computer Vision 600.461/661 Assignment 2
	% Args:
	%	feature_coords1 : list of (row,col) feature coordinates from image1
	%	feature_coords2 : list of (row,col)feature coordinates from image2
	% 	image1 : The input image corresponding to features_coords1
	% 	image2 : The input image corresponding to features_coords2
	% Returns:
	% 	matches : list of index pairs of possible matches. For example, if the 4-th feature in feature_coords1 and the 1-st feature
	%							  in feature_coords2 are determined to be matches, the list should contain (4,1).
	%%%

    % Side by side color image for visualization in the end
    [h1, w1, ~] = size(image1);
    [h2, w2, ~] = size(image2);
    if h1 ~= h2 || w1 ~= w2
        max_h = max(h1,h2);
        max_w = max(w1,w2);
        image1 = padarray(image1, [(max_h - h1) (max_w - w1)], 'post');
        image2 = padarray(image2, [(max_h - h2) (max_w - w2)], 'post');
    end
    
    SbS = [image1 image2];

    image1 = double(rgb2gray(image1));
    image2 = double(rgb2gray(image2));
    
    % Make Z(x, y) for image1
    image1_mean = mean2(image1);
    image1_z = zeros(size(image1));
    [num_rows1, num_cols1] = size(image1);
    for r = 1:num_rows1
        for c = 1:num_cols1
            image1_z(r, c) = image1(r, c) - image1_mean;
        end
    end
    
    % Make Z(x, y) for image2
    image2_mean = mean2(image2);
    image2_z = zeros(size(image2));
    [num_rows2, num_cols2] = size(image2);
    for r = 1:num_rows2
        for c = 1:num_cols2
            image2_z(r, c) = image2(r, c) - image2_mean;
        end
    end
    
    % Make ZN(x, y) for image1
    image1_std = std2(image1_z);
    image1_zn = zeros(size(image1));
    for r = 1:num_rows1
        for c = 1:num_cols1
            image1_zn(r, c) = image1_z(r, c)./image1_std;
        end
    end
    
    % Make ZN(x, y) for image2
    image2_std = std2(image2_z);
    image2_zn = zeros(size(image2));
    for r = 1:num_rows2
        for c = 1:num_cols2
            image2_zn(r, c) = image2_z(r, c)./image2_std;
        end
    end

    % Compute matches for image1
    window_size = 7;
    matches_1 = [];
    [num_features1, ~] = size(feature_coords1);
    [num_features2, ~] = size(feature_coords2);
    for i = 1:num_features1
        r1 = feature_coords1(i, 1);
        c1 = feature_coords1(i, 2);
        max_r2 = feature_coords2(1, 1);
        max_c2 = feature_coords2(1, 2);
        max_i2 = 0;
        reach = (window_size - 1) / 2;
        window_1 = zeros(window_size, window_size);
        [image_rows, image_cols] = size(image1_zn);
        if r1 - reach > 0 && r1 + reach <= image_rows && c1 - reach > 0 && c1 + reach <= image_cols
            window_1(1:window_size, 1:window_size) = image1_zn(r1 - reach: r1 + reach, c1 - reach: c1 + reach);
        else
            window_1(1:window_size, 1:window_size) = zeros(window_size, window_size);
        end
        n = window_size.^2;
        max_ncc = -1;
        for j = 1:num_features2
            r2 = feature_coords2(j, 1);
            c2 = feature_coords2(j, 2);
            window_2 = [];
            [image_rows, image_cols] = size(image2_zn);
            if r2 - reach > 0 && r2 + reach <= image_rows && c2 - reach > 0 && c2 + reach <= image_cols
                window_2(1:window_size, 1:window_size) = image2_zn(r2 - reach: r2 + reach, c2 - reach: c2 + reach);
            else
                window_2(1:window_size, 1:window_size) = zeros(window_size, window_size);
            end
            curr_ncc = 0;
            for r = 1:window_size
                for c = 1:window_size
                    curr_ncc = curr_ncc + window_1(r, c) * window_2(r, c);
                end
            end
            curr_ncc = curr_ncc./n;
            if curr_ncc > max_ncc
                max_ncc = curr_ncc;
                max_r2 = r2;
                max_c2 = c2;
                max_i2 = j;
            end
        end
        matches_1 = [matches_1; max_r2 max_c2 max_i2];
    end
    
    % Compute matches for image2
    window_size = 5;
    matches_2 = [];
    for i = 1:num_features2
        r2 = feature_coords2(i, 1);
        c2 = feature_coords2(i, 2);
        max_r1 = feature_coords1(1, 1);
        max_c1 = feature_coords1(1, 2);
        max_i1 = 0;
        reach = (window_size - 1) / 2;
        window_2 = zeros(window_size, window_size);
        [image_rows, image_cols] = size(image2_zn);
        if r2 - reach > 0 && r2 + reach <= image_rows && c2 - reach > 0 && c2 + reach <= image_cols
            window_2(1:window_size, 1:window_size) = image2_zn(r2 - reach: r2 + reach, c2 - reach: c2 + reach);
        else
            window_2(1:window_size, 1:window_size) = zeros(window_size, window_size);
        end
        window_2(1:window_size, 1: window_size) = image2_zn(r2 - reach: r2 + reach, c2 - reach: c2 + reach);
        n = window_size.^2;
        max_ncc = -1;
        for j = 1:num_features1
            r1 = feature_coords1(j, 1);
            c1 = feature_coords1(j, 2);
            window_1 = zeros(window_size, window_size);
            [image_rows, image_cols] = size(image1_zn);
            if r1 - reach > 0 && r1 + reach <= image_rows && c1 - reach > 0 && c1 + reach <= image_cols
                window_1(1:window_size, 1:window_size) = image1_zn(r1 - reach: r1 + reach, c1 - reach: c1 + reach);
            else
                window_1(1:window_size, 1:window_size) = zeros(window_size, window_size);
            end
            curr_ncc = 0;
            for r = 1:window_size
                for c = 1:window_size
                    curr_ncc = curr_ncc + window_2(r, c) * window_1(r, c);
                end
            end
            curr_ncc = curr_ncc./n;
            if curr_ncc > max_ncc
                max_ncc = curr_ncc;
                max_r1 = r1;
                max_c1 = c1;
                max_i1 = j;
            end
        end
        matches_2 = [matches_2; max_r1 max_c1 max_i1];
    end

    % Compute final matches
    matches = [];
    for i = 1:num_features1
       mi2 = matches_1(i, 3);
       mi1 = matches_2(mi2, 3);
       if mi1 == i
           matches = [matches; i mi2];
       end
    end
    
    [~, num_cols] = size(image1); 
    
    % Draw matches on SbS
    [num_matches, ~] = size(matches);
    figure(1), imshow(SbS,[]), hold on 
    for m = 1:num_matches
        m1 = matches(m, 1);
        r1 = feature_coords1(m1, 1);
        c1 = feature_coords1(m1, 2);
        m2 = matches(m, 2);
        r2 = feature_coords2(m2, 1);
        c2 = feature_coords2(m2, 2) + num_cols;
        p1 = [r1,c1];
        p2 = [r2,c2];
        plot([p1(2),p2(2)],[p1(1),p2(1)],'Color','g','LineWidth',1)
    end
    hold off
end

