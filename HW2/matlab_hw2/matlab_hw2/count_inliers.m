function [num_inliers, inliers, outliers] = count_inliers(matches, r_diff, c_diff, features1, features2)
    [r, ~] = size(matches);
    num_inliers = 0;
    degree_of_acceptance = 0.1;
    inliers = [];
    outliers = [];
    for i = 1:r
        match_im1_index = matches(i, 1);
        match_im2_index = matches(i, 2);
        feature_im1_r = features1(match_im1_index, 1);
        feature_im1_c = features1(match_im1_index, 2);
        feature_im2_r = features2(match_im2_index, 1);
        feature_im2_c = features2(match_im2_index, 2);
        [curr_r_diff, curr_c_diff] = compute_t(feature_im1_r, feature_im1_c, feature_im2_r, feature_im2_c);
        if (curr_r_diff * (1 - degree_of_acceptance) <= r_diff || r_diff <= curr_r_diff * (1 + degree_of_acceptance)) && (curr_c_diff * (1 - degree_of_acceptance) <= c_diff || c_diff <= curr_c_diff * (1 + degree_of_acceptance))
            num_inliers = num_inliers + 1;
            inliers = [inliers; match_im1_index match_im2_index];
        else
            outliers = [outliers; match_im1_index match_im2_index];
        end
    end
end