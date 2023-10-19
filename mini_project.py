import cv2, numpy as np, random
from ransac import ransac_homography


img1 = cv2.imread("media2/img1.jpg")
img2 = cv2.imread("media2/img2.jpg")

scale = 500/img1.shape[0]
img1=cv2.resize(img1,None, fx=scale, fy=scale)
img2=cv2.resize(img2,None, fx=scale, fy=scale)

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
matches = [kp_1[m[0].queryIdx].pt + kp_2[m[0].trainIdx].pt for m in good]

# Showing both images with matches and correspondences
matched_image = cv2.drawMatchesKnn(img1, kp_1, img2, kp_2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow("matched_image", matched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Here we calculate H with ransac:
h = ransac_homography(matches, n_samples=4, tolerance=1, max_iterations=5000, threshold=0.9)

# Warping image 1 to stich it later.
img_warped = cv2.warpPerspective(img1, h, (img2.shape[1],img2.shape[0]))

alpha = 0.4
result = cv2.addWeighted(img_warped, alpha, img2, 1-alpha, 0)


# Putting together all the images so they can be shown with just one winow
hor1 = np.concatenate((img1, img2), axis=1) 
hor2 = np.concatenate((img_warped, result), axis=1) 
subplot = np.concatenate((hor1,hor2), axis=0)

subplot=cv2.resize(subplot,None, fx=0.75, fy=0.75)
cv2.imshow("blend of warpe  d + img2", subplot)

# Creating panoramic image with img1 + img2.
result2 = cv2.warpPerspective(img1, h, (img1.shape[1] + img2.shape[1], int(img1.shape[0]*1.25)))
result2[0:img2.shape[0], 0:img2.shape[1]] = img2
cv2.imshow("Panoramic img1 + img2", result2)

cv2.waitKey(0)
cv2.destroyAllWindows()