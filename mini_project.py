import cv2, numpy as np
from os import listdir
from ransac import ransac_homography


def read_images(file):

    image_names = listdir(file)

    images = []

    for name in image_names:
        img = cv2.imread(file+name)

        scale = 500/img.shape[0]
        img=cv2.resize(img,None, fx=scale, fy=scale)

        images.append(img)
    

    return images[0], images[1:]

def imgs_feature_matching(img1,img2, info=False):

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # SIFT keypoints/descriptors:
    sift = cv2.SIFT_create()
    kp_1, des_1 = sift.detectAndCompute(img1_gray,None)
    kp_2, des_2 = sift.detectAndCompute(img2_gray,None)

    # Matching keypoints:
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des_1, des_2, k=2)

    # Filtering matches:
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    # Filtering keypoints:
    matches = np.array([kp_1[m[0].queryIdx].pt + kp_2[m[0].trainIdx].pt for m in good])

    # Showing both images with matches and correspondences
    matched_image = cv2.drawMatchesKnn(img1, kp_1, img2, kp_2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    if info:
        cv2.imshow("matched_image", matched_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return matches

def warpTwoImages(img1, img2, H):
    '''warp img2 to img1 with homograph H'''
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate

    warped_image = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))


    blank_image = np.zeros(warped_image.shape, np.uint8)
    blank_image[t[1]:h1+t[1],t[0]:w1+t[0]] = img1


    img_gray = cv2.cvtColor(blank_image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(img_gray, 0,255, type=cv2.THRESH_BINARY_INV)

    warped_image = cv2.bitwise_and(warped_image, warped_image, mask=mask)

    result = cv2.add(blank_image, warped_image)

    return result

def get_information_cv2Homography(matches):

    # First obtain the homography from cv2.
    matches_src, matches_dst = np.split(np.array(matches), 2, 1)
    h, mask = cv2.findHomography(matches_src, matches_dst, cv2.RANSAC)

    mask = np.array([bool(m) for m in mask]) # Change mask from 0-1 to False-True.

    # Next obtain the proyection for each source point and calculate the error.
    n_inliers = len(mask[mask])

    src, dst = np.split(np.array(matches).transpose(), 2, 0)     
    src = np.insert(src,2,1,axis=0)     # Adding 1s at the end of the point (2D -> 3D point)
    dst = np.insert(dst,2,1,axis=0)

    src_proy = np.dot(h,src)            # Multiplying all the 3D points for the homography.
    src_proy /= src_proy[-1]            # Normalizing proyected points.

    errors = np.linalg.norm(dst-src_proy, axis=0) # Errors: distance between destination position and proyected position.
    
    error_inliers = errors[mask]

    mse = np.square(error_inliers).mean()
    
    print(f"[#] Result h2: {n_inliers}/{len(matches)}: {n_inliers/len(matches):.3f} ratio. MSE: {mse:.3f}\n")

    return h



if __name__ == "__main__":

    base_image, images = read_images("media5/")


    for image in images[:]:
        matches = imgs_feature_matching(image, base_image, info=False)

        # Here we calculate H with ransac:
        h1 = ransac_homography(matches, n_samples=4, tolerance=1, max_iterations=2100, threshold=0.6, info=True)

        h2 = get_information_cv2Homography(matches)

        base_image = warpTwoImages(base_image, image, h1)

        #print(f"{h1 - h2}")


    # Resize and show the final result:
    scale = 700/base_image.shape[0]
    base_image = cv2.resize(base_image, None, fx=scale, fy=scale)
    cv2.imshow("Panoramic img1 + img2", base_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()