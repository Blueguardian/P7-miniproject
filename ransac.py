import cv2, random, numpy as np

def ransac(matches, tolerance, max_iterations, threshold):

    best_inliers = 0
    best_H = np.eye(3,3)

    for i in range(max_iterations):

        samples = np.array(random.sample(matches, 4))

        samp_src = samples[:,:2]
        samp_dst = samples[:, 2:]

        H, _ = cv2.findHomography(samp_src, samp_dst, 0) # We will have to change this and do it from scratch.

        inliers=0

        for pair in matches:

            p1 = np.append(pair[:2],1).reshape(3,1)
            p2 = np.append(pair[2:],1).reshape(3,1)

            p2_aprox = np.dot(H,p1)

            dist = np.linalg.norm(p2 - p2_aprox)

            if dist < tolerance:
                inliers += 1


        if inliers > best_inliers:
            best_inliers = inliers
            best_H = H

        if best_inliers/len(matches) >= threshold: 
            print(f"n of iterations: {i+1}")
            break

        
        #print(f"{inliers}/{len(matches)}: \tbest: {best_inliers}")

    print(f"\n[{best_inliers}/{len(matches)}]: {best_inliers/len(matches)}% ratio")

    return best_H