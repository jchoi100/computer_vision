function [ ] = hw2(image1_file_name, image2_file_name)
    image1 = imread(image1_file_name);
    image2 = imread(image2_file_name);
    [r1, c1] = detect_features(image1);
    [r2, c2] = detect_features(image2);
    feature_coords1 = [];
    for r = 1:numel(r1)
        feature_coords1 = [feature_coords1; r1(r) c1(r)];
    end
    feature_coords2 = [];
    for r = 1:numel(r2)
        feature_coords2 = [feature_coords2; r2(r) c2(r)];
    end
    matches = match_features(feature_coords1, feature_coords2, image1, image2);
    affine_xform = compute_affine_xform(matches,feature_coords1,feature_coords2,image1,image2);
end

