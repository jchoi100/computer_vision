function [r_diff, c_diff] = compute_t(feature_im1_r, feature_im1_c, feature_im2_r, feature_im2_c)
    r_diff = feature_im1_r - feature_im2_r;
    c_diff = feature_im1_c - feature_im2_c;
end