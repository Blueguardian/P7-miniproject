import numpy as np, random

def get_homography(src, dst):

    A = []
    N = len(src) # A matrix can be ajusted for N_points >= 4.

    for i in range(N):
        x, y = src[i][0], src[i][1]
        xp, yp = dst[i][0], dst[i][1]
        A.append([x, y, 1, 0, 0, 0, -x * xp, -xp * y, -xp])
        A.append([0, 0, 0, x, y, 1, -yp * x, -yp * y, -yp])

    A = np.asarray(A)

    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1, :] / Vh[-1, -1]

    H = L.reshape(3, 3)
    H /= H[2, 2]            # Normalization step. H is [3, 3] matrix.

    return H

def ransac_homography(matches, n_samples=4, tolerance=2, max_iterations=500, threshold=1):

    best_inliers = 0
    best_H = np.eye(3)
    best_MSE = np.inf

    # separate source points from destination points.
    matches_src, matches_dst = np.split(np.array(matches).transpose(), 2, 0)     
    matches_src = np.insert(matches_src,2,1,axis=0)     # Adding 1 at the end and reshaping to columns instead of rows.
    matches_dst = np.insert(matches_dst,2,1,axis=0) 


    for i in range(max_iterations):

        # First we get n random samples from all the matches.
        samples = np.array(random.sample(matches, n_samples))
        samp_src, samp_dst = np.split(samples, 2, 1)

        H = get_homography(samp_src,samp_dst)   # Obtaining the Homography matrix.                        

        # Proyecting all the source points into the destination space using the Homography.
        src_proy = np.dot(H,matches_src)

        # Obtaining the distances between all the proyected points and their correspondences. 
        distances = np.linalg.norm(matches_dst-src_proy,axis=0)

        # Now we the number of inliers for this H and the Mean Squared Error of the distances between matches.
        error_inliers = distances[distances<=tolerance]
        n_inliers = len(error_inliers)
        if n_inliers:
            mse = np.square(error_inliers).mean()
        else:
            mse = np.inf
        
        if n_inliers > best_inliers or (n_inliers == best_inliers and mse < best_MSE):
            best_inliers = n_inliers
            best_H = H
            best_MSE = mse

        if (i+1) % 500 == 0:
            print(f"-> i: {i+1} \tbest: [{best_inliers}/{len(matches)}] \tMSE: {best_MSE:.3f}")

        if best_inliers/len(matches) >= threshold:
            break


    print(f"\n[#]: n of iterations done: {i+1}.")
    print(f"[#] Result: {best_inliers}/{len(matches)}: {best_inliers/len(matches):.3f} ratio. MSE: {best_MSE:.3f}\n")

    return best_H