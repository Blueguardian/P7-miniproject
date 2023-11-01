import cv2, numpy as np
from mini_project import *

np.set_printoptions(suppress=True)

def get_cv2_Homography(matches, tolerance=1, maxIter=1000, info=False):

    # First obtain the homography from cv2.
    matches_src, matches_dst = np.split(np.array(matches), 2, 1)
    h, _ = cv2.findHomography(matches_src, matches_dst, cv2.RANSAC, ransacReprojThreshold=tolerance, maxIters=maxIter)

    # Next obtain the proyection for each source point and calculate the error.
    src, dst = np.split(np.array(matches).transpose(), 2, 0)     
    src = np.insert(src,2,1,axis=0)     # Adding 1s at the end of the point (2D -> 3D point)
    dst = np.insert(dst,2,1,axis=0)

    src_proy = np.dot(h,src)            # Multiplying all the 3D points for the homography.
    src_proy /= src_proy[-1]            # Normalizing proyected points.

    errors = np.linalg.norm(dst-src_proy, axis=0) # Errors: distance between destination position and proyected position.
    mask = errors <= tolerance
    error_inliers = errors[mask]

    n_inliers = len(mask[mask])

    mse = np.square(error_inliers).mean()
    
    if info:
        print(f"[#] Result h2: {n_inliers}/{len(matches)}: {n_inliers/len(matches):.3f} ratio. MSE: {mse:.3f}")

    return h

def transform_image(img, theta=0,tx=0,ty=0,s=1):
    theta = np.deg2rad(theta)

    H = np.array([  [np.cos(theta), -np.sin(theta), tx],
                    [np.sin(theta),  np.cos(theta), ty],
                    [      -0.0005,              0,  s]], dtype=np.float32)


    warped_image = cv2.warpPerspective(img, H, (img.shape[1]-400, img.shape[0]))


    return np.linalg.inv(H), warped_image

def compare_results(matches, h_sol, H, tolerance=1):

    matches_src, matches_dst = np.split(matches.transpose(), 2, 0)
    matches_src = np.insert(matches_src,2,1,axis=0)     # Adding 1s at the end of the point (2D -> 3D point)
    matches_dst = np.insert(matches_dst,2,1,axis=0)     # and reshaping points in columns instead of rows.

    # Proyecting all the source points into the destination space using the Real Homography.
    real_src_proy = np.dot(h_sol, matches_src)
    real_src_proy /= real_src_proy[-1]                # Normalizing the points.

    real_errors = np.linalg.norm(matches_dst-real_src_proy, axis=0)
    real_mask = real_errors <= tolerance


    # Proyecting all the source points into the destination space using the Homography.
    src_proy = np.dot(H , matches_src)
    src_proy /= src_proy[-1]                # Normalizing the points.

    # Obtaining the approximated distance between all the proyected points and their correspondences.
    exp_errors = np.linalg.norm(matches_dst-src_proy, axis=0)
    exp_mask = exp_errors <= tolerance


    ### Now we obtain the data:


    # Number of expected inliers.
    n_exp_inliers = len(exp_errors[exp_mask])

    # Number of expected outlier.
    n_exp_outliers = len(matches)-n_exp_inliers

    # Number of real inliers actually found by H.
    real_mask_found = exp_mask[real_mask] # mask of inliers and outliers
    n_real_inliers_found = len(real_mask_found[real_mask_found])

    # Expected inliers MSE:
    exp_inliers_errors = exp_errors[exp_mask]
    exp_inliers_mse = np.square(exp_inliers_errors).mean()

    # Real inliers MSE using H:
    filtered_exp_errors = exp_errors[real_mask]             # Errors of only the points considered as real inliers.
    filtered_exp_inliers_mse = np.square(filtered_exp_errors).mean() # MSE of real inliers with another H.

    return n_exp_inliers, n_exp_outliers, n_real_inliers_found, exp_inliers_mse, filtered_exp_inliers_mse


if __name__ == "__main__":

    base_image = cv2.imread("test_media/img1.jpg")
    tol = 1

    h_sol, warped_img = transform_image(base_image, 100,1000,-600, 2)
    cv2.imshow("warped img", warped_img)

    # Run detector an descriptor to find matches between both images.
    matches = imgs_feature_matching(warped_img, base_image, threshold=0.75, info=False)

    # Here we calculate H with our ransac:
    h1 = ransac_homography(matches, tolerance=tol, threshold=1, info=True)

    # Here we calculate H with openCV ransac:
    h2 = get_cv2_Homography(matches, tolerance=tol, info=True)
    print()


    inliers, outliers, inliers_found, exp_mse, filtered_mse = compare_results(matches, h_sol, h1, tolerance=tol)
    print(f"[h1] {inliers=}, {outliers=}, {inliers_found=}, {exp_mse=:.2f}, {filtered_mse=:.2f}")

    inliers, outliers, inliers_found, exp_mse, filtered_mse = compare_results(matches, h_sol, h2, tolerance=tol)
    print(f"[h2] {inliers=}, {outliers=}, {inliers_found=}, {exp_mse=:.2f}, {filtered_mse=:.2f}")

    inliers, outliers, inliers_found, exp_mse, filtered_mse = compare_results(matches, h_sol, h_sol, tolerance=tol)
    print(f"[h0] {inliers=}, {outliers=}, {inliers_found=}, {exp_mse=:.2f}, {filtered_mse=:.2f}\n")


    print("Real homography matrix:\n", np.round(h_sol,2))
    print("Homography matrix calculated with our method:\n", np.round(h1, 2))
    print("Homography matrix calculatedh with cv2 method:\n",np.round(h2, 2))


    # Warping images:
    fixed_img_approx = cv2.warpPerspective(warped_img, h1, base_image.shape[1::-1])
    fixed_img_real = cv2.warpPerspective(warped_img, h_sol, base_image.shape[1::-1])

    # Show the final result:
    cv2.imshow("fixed image real", fixed_img_real)
    cv2.imshow("fixed image approximated", fixed_img_approx)

    cv2.waitKey(0)
    cv2.destroyAllWindows()