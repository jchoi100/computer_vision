function [affine_xform] = compute_affine_xform(matches,features1,features2,image1,image2)
	%%%
	% Computer Vision 600.461/661 Assignment 2
	% Args:
	%	matches : list of index pairs of possible matches. For example, if the 4-th feature in feature_coords1 and the 1-st feature
	%							  in feature_coords2 are determined to be matches, the list should contain (4,1).
    %   features1 (list of tuples) : list of feature coordinates corresponding to image1
    %   features2 (list of tuples) : list of feature coordinates corresponding to image2
	% 	image1 : The input image corresponding to features_coords1
	% 	image2 : The input image corresponding to features_coords2
	% Returns:
	%	affine_xform (ndarray): a 3x3 Affine transformation matrix between the two images, computed using the matches.
	% 
    
    % Side by side color image for visualization in the end
    [h1, w1, ~] = size(image1);
    [h2, w2, ~] = size(image2);
    if h1 ~= h2 || w1 ~= w2
        max_h = max(h1,h2);
        max_w = max(w1,w2);
        image1 = padarray(image1, [(max_h - h1) (max_w - w1)], 'post');
        image2 = padarray(image2, [(max_h - h2) (max_w - w2)], 'post');
    end
    
    [~, num_cols, ~] = size(image1); 
    SbS = [image1 image2];
    [num_matches, ~] = size(matches);
    ransac_thresh = 15;
    max_num_inliers = 0;
    best_inlier_indices = [];
    
    for i = 1:500
        r1 = randi(num_matches,1,3);
        
        match1_im1_index = matches(r1(1), 1);
        match1_im2_index = matches(r1(1), 2);
        
        match2_im1_index = matches(r1(2), 1);
        match2_im2_index = matches(r1(2), 2);
        
        match3_im1_index = matches(r1(3), 1);
        match3_im2_index = matches(r1(3), 2);
        
        y1  = features1(match1_im1_index, 1);
        x1  = features1(match1_im1_index, 2);
        y1p = features2(match1_im2_index, 1);
        x1p = features2(match1_im2_index, 2);
        y2  = features1(match2_im1_index, 1);
        x2  = features1(match2_im1_index, 2);
        y2p = features2(match2_im2_index, 1);
        x2p = features2(match2_im2_index, 2);
        y3  = features1(match3_im1_index, 1);
        x3  = features1(match3_im1_index, 2);
        y3p = features2(match3_im2_index, 1);
        x3p = features2(match3_im2_index, 2);
        
        A = [x1 y1 1  0  0 0; 
              0  0 0 x1 y1 1; 
             x2 y2 1  0  0 0; 
              0  0 0 x2 y2 1;
             x3 y3 1  0  0 0;
              0  0 0 x3 y3 1;];
        b = [x1p;y1p;x2p;y2p;x3p;y3p];
        x = linsolve(A, b);
        T = [x(1) x(2) x(3); x(4) x(5) x(6); 0 0 1];
        
        curr_error = 0;
        num_inliers = 0;
        inlier_indices=[];
        for j = 1:num_matches
            image1_match_index = matches(j, 1);
            image2_match_index = matches(j, 2);
            y_test1 = features1(image1_match_index, 1);
            x_test1 = features1(image1_match_index, 2);
            y_test2 = features2(image2_match_index, 1);
            x_test2 = features2(image2_match_index, 2);
            test_match_image1_tr = T * [x_test1;y_test1;1];
            err = sqrt((test_match_image1_tr(1) - x_test2)^2 + (test_match_image1_tr(2) - y_test2)^2);
            if err < ransac_thresh
                num_inliers = num_inliers + 1;
                inlier_indices = [inlier_indices;j];
            end
            curr_error  = curr_error + err;
        end
        
        if num_inliers > max_num_inliers
            max_num_inliers = num_inliers;
            best_inlier_indices = inlier_indices;
        end
    end
    
    A = [];
    b = [];
    image1_inlier_rs = [];
    image1_inlier_cs = [];

    for ind = 1:size(best_inlier_indices,1)
       idx=best_inlier_indices(ind);
       match1_im1_index = matches(idx, 1);
       match1_im2_index = matches(idx, 2);
       y1  = features1(match1_im1_index, 1);
       x1  = features1(match1_im1_index, 2);
       y1p = features2(match1_im2_index, 1);
       x1p = features2(match1_im2_index, 2);
       image1_inlier_rs = [image1_inlier_rs; y1];
       image1_inlier_cs = [image1_inlier_cs; x1];
       A = [A;
            x1 y1 1  0  0 0; 
             0 0  0 x1 y1 1];
       b = [b; x1p ; y1p];
    end
        
    x = linsolve(A, b);
    affine_xform = [x(1) x(2) x(3); x(4) x(5) x(6); 0 0 1];
        
    tform = affine2d(affine_xform');
    image1_warped = imwarp(image1, tform,'OutputView',imref2d(size(image2)));
    out = imfuse(image2,image1_warped,'blend','Scaling','joint');
    figure
    imshow(out,[])
       
    % Draw matches on SbS
    [num_matches, ~] = size(matches);
    figure(1), imshow(SbS,[]), hold on 
    for m = 1:num_matches
        m1 = matches(m, 1);
        r1 = features1(m1, 1);
        c1 = features1(m1, 2);
        m2 = matches(m, 2);
        r2 = features2(m2, 1);
        c2 = features2(m2, 2) + num_cols;
        p1 = [r1,c1];
        p2 = [r2,c2];
        if ismember(r1, image1_inlier_rs) && ismember(c1, image1_inlier_cs)
            plot([p1(2),p2(2)],[p1(1),p2(1)],'Color','g','LineWidth',1)
        else
            plot([p1(2),p2(2)],[p1(1),p2(1)],'Color','r','LineWidth',1)
        end
    end
    hold off
    
end