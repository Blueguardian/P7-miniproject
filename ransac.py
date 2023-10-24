import numpy as np, random

def get_homography(samples):

    src, dst = np.split(np.array(samples), 2, 1)

    A = []
    N = len(src) # A matrix can be ajusted for N_points >= 4.

    for i in range(N):
        xs, ys  = src[i]
        xd, yd  = dst[i]

        A.append([-xs, -ys, -1,  0 ,  0 ,  0, xs * xd, xd * ys, xd])
        A.append([0  ,  0 ,  0, -xs, -ys, -1, yd * xs, yd * ys, yd])

    A = np.asarray(A)

    U, S, Vh = np.linalg.svd(A) # Calculate SVD of A to solve A*h = 0. Whithin the last row of V will be H.

    H = np.reshape(Vh[8], (3, 3)) # Reshape the min singular value into a 3 by 3 matrix

    H = H / H.item(8) # Normalize to get H

    return H

def is_non_collinear(points_matrix):
    p = np.append(points_matrix, points_matrix[:2],axis=0)

    for i in range(4):
        temp = p[i:3+i]

        rank = np.linalg.matrix_rank(temp)
        if rank < 2: 
            return False

    return True

def ransac_homography(matches, n_samples=4, tolerance=1, max_iterations=500, threshold=1, info=False):

    best_inliers = 0
    best_H = np.eye(3)
    best_MSE = np.inf

    best_errors = []

    # separate source points from destination points.
    matches_src, matches_dst = np.split(matches.transpose(), 2, 0)     
    matches_src = np.insert(matches_src,2,1,axis=0)     # Adding 1s at the end of the point (2D -> 3D point)
    matches_dst = np.insert(matches_dst,2,1,axis=0)     # and reshaping points in columns instead of rows.


    for i in range(max_iterations): 

        # First we get n random samples from all the matches.
        samples = random.sample(matches.tolist(), n_samples)

        H = get_homography(samples)   # Obtaining the Homography matrix.                        

        # Proyecting all the source points into the destination space using the Homography.
        src_proy = np.dot(H,matches_src)
        src_proy /= src_proy[-1]            # Normalizing the points.

        # Obtaining the distances between all the proyected points and their correspondences. 
        errors = np.linalg.norm(matches_dst-src_proy, axis=0)

        # Now we the number of inliers for this H and the Mean Squared Error of the error between matches.
        error_inliers = errors[errors<=tolerance]
        n_inliers = len(error_inliers)

        if n_inliers:
            mse = np.square(error_inliers).mean()
        else:
            mse = np.inf
        
        if n_inliers > best_inliers or (n_inliers == best_inliers and mse < best_MSE):
            best_errors = errors <= tolerance
            best_inliers = n_inliers
            best_MSE = mse
            best_H = H


        if info and (i+1) % 2000 == 0:
            print(f"-> i: {i+1} \tbest: [{best_inliers}/{len(matches)}] \tMSE: {best_MSE:.3f}")

        if best_inliers/len(matches) >= threshold:
            break

    ### Now the homography is refined by recalculating it but this time with all the inliers:

    inliers = matches[best_errors] # Getting only the inliers.

    refined_H = get_homography(inliers)   # Obtaining the Homography matrix with all the inliers.                        

    # Proyecting all the source points into the destination space using the Homography.
    src_proy = np.dot(refined_H, matches_src)
    src_proy /= src_proy[-1]                    # Normalizing the points.

    # Obtaining the distances between all the proyected points and their correspondences. 
    errors = np.linalg.norm(matches_dst-src_proy, axis=0)

    # Number of inliers for this refined_H and the Mean Squared Error of the distances between matches.
    error_inliers = errors[errors<=tolerance]
    refined_n_inliers = len(error_inliers)
    refined_mse = np.square(error_inliers).mean()


    if info:
        print(f"\n[#]: n of iterations done: {i+1}.")
        print(f"[#] Result: {best_inliers}/{len(matches)}: {best_inliers/len(matches):.3f} ratio. MSE: {best_MSE:.3f}")
        print(f"[#] Refined result: {refined_n_inliers}/{len(matches)}: {refined_n_inliers/len(matches):.3f} ratio. MSE: {refined_mse:.3f}\n")

 

    return refined_H