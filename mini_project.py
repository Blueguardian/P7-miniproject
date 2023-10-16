import cv2
from ransac import ransac

img1 = cv2.imread("media2/img1.jpg")
img2 = cv2.imread("media2/img2.jpg")

img1=cv2.resize(img1,None, fx=0.4, fy=0.4)
img2=cv2.resize(img2,None, fx=0.4, fy=0.4)

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

matched_image = cv2.drawMatchesKnn(img1, kp_1, img2, kp_2, good, None)
cv2.imshow("matched_image", matched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# a partir de aqui hacemos el ransac
h = ransac(matches, tolerance=3, max_iterations=500, threshold=0.9)


img_warped = cv2.warpPerspective(img1, h, (img2.shape[1],img2.shape[0]))


# Print transformation matrix and images.
print("\n",h,"\n")

cv2.imshow("img1", img1)
cv2.imshow("img2", img2)
cv2.imshow("img1 warped", img_warped)
cv2.waitKey(0)


cv2.destroyAllWindows()