function [affine_xform] = compute_affine_xform(matches,features1,features2,image1,image2)
    %%%
    % Computer Vision 600.461/661 Assignment 2
    % Args:
    %   matches : list of index pairs of possible matches. For example, if the 4-th feature in feature_coords1 and the 1-st feature
    %                             in feature_coords2 are determined to be matches, the list should contain (4,1).
    %   features1 (list of tuples) : list of feature coordinates corresponding to image1
    %   features2 (list of tuples) : list of feature coordinates corresponding to image2
    %   image1 : The input image corresponding to features_coords1
    %   image2 : The input image corresponding to features_coords2
    % Returns:
    %   affine_xform (ndarray): a 3x3 Affine transformation matrix between the two images, computed using the matches.
    % 
    
    max_num_inliers = 0;
    [num_iterations, dummy] = size(matches);
    [num_matches, dummy] = size(matches);
    best_feature_im1_r = 0;
    best_feature_im1_c = 0;
    best_feature_im2_r = 0;
    best_feature_im2_c = 0;
    best_inliers = [];
    best_outliers = [];
    
    for i = 1:num_iterations
        r1 = randi(num_matches,1,6)
        
        
        random_index = randi([1 num_matches], 1, 1);
        match_im1_index = matches(random_index, 1);
        match_im2_index = matches(random_index, 2);
        feature_im1_r = features1(match_im1_index, 1);
        feature_im1_c = features1(match_im1_index, 2);
        feature_im2_r = features2(match_im2_index, 1);
        feature_im2_c = features2(match_im2_index, 2);
        [r_diff, c_diff] = compute_t(feature_im1_r, feature_im1_c, feature_im2_r, feature_im2_c);
        for j = 1:num_matches
            [num_inliers, inliers, outliers] = count_inliers(matches, r_diff, c_diff, features1, features2);
            if num_inliers > max_num_inliers
                max_num_inliers = num_inliers;
                best_feature_im1_r = feature_im1_r;
                best_feature_im1_c = feature_im1_c;
                best_feature_im2_r = feature_im2_r;
                best_feature_im2_c = feature_im2_c;
                best_inliers = inliers;
                best_outliers = outliers;
            end
        end
    end
    
    best_feature_im1_r;
    best_feature_im1_c;
    best_feature_im2_r;
    best_feature_im2_c;
    best_inliers;
    best_outliers;
    
    [num_inliers, dummy] = size(best_inliers);
    A = [];
    for i = 1:num_inliers
       image1_inlier_index = best_inliers(i, 1);
       image2_inlier_index = best_inliers(i, 2);
       image1_inlier_r = features1(image1_inlier_index, 1);
       image1_inlier_c = features1(image1_inlier_index, 2);
       image2_inlier_r = features2(image2_inlier_index, 1);
       image2_inlier_c = features2(image2_inlier_index, 2);
       A = [A; image1_inlier_r image1_inlier_c 1 0 0 0 -image2_inlier_r*image1_inlier_r -image2_inlier_r*image1_inlier_c -image2_inlier_r];
       A = [A; 0 0 0 image1_inlier_r image1_inlier_c 1 -image2_inlier_c*image1_inlier_r -image2_inlier_c*image1_inlier_c -image2_inlier_c];
    end
    
    [V, D] = eig(A'*A);
    [num_rows, num_cols] = size(D);
    eigen_value = 1/0;
    eigen_vector = [];
    for r = 1:num_rows
        for c = 1:num_cols
            if D(r, c) < eigen_value && D(r, c) ~= 0
                eigen_value = D(r, c);
                eigen_vector = V(:,c);
            end
        end
    end
    eigen_vector = eigen_vector';
    h = [eigen_vector(1:3); eigen_vector(4:6); eigen_vector(7:9);];
    tform = affine2d(h);
    image2_warped = imwarp(image2, tform);
    imshow(image2_warped,[])
    % Found the best match to overlap on.
    % TODO: Do affine transform on one of the images first before finding matches....
    
end