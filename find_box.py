import cv2
import numpy as np

def show(name, img) :
    cv2.imshow(name,img) #displays the graphical image
    cv2.waitKey(0)
    cv2.destroyAllWindows()

obj = cv2.imread("Degradation.jpeg",cv2.IMREAD_GRAYSCALE) #this returns a numpy array, pythons library for grids of numbers of the same type
scene = cv2.imread("scene.jpg",cv2.IMREAD_GRAYSCALE)
if obj is None or scene is None : 
    raise ValueError("Wrong Image file/No Image file. Check the folder")

print("object:", obj.shape, obj.dtype)
print("scene:",scene.shape,scene.dtype)
show("object",obj)

orb= cv2.ORB_create(2000)
kp1,des1 = orb.detectAndCompute(obj,None)
kp2, des2 = orb.detectAndCompute(scene,None)

print("keypoints:",len(kp1),len(kp2))
print("descriptors : ", des1.shape,des1.dtype)
show("keypoints",cv2.drawKeypoints(obj,kp1,None, color = (0,255,0)))

bf = cv2.BFMatcher(cv2.NORM_HAMMING)
pairs =  bf.knnMatch(des1, des2, k=2 )
good =  [m for m, n in pairs if m.distance < 0.75 * n.distance]
print("candidates: ",len(pairs),"after this ratio test:",len(good))

vis = cv2.drawMatches(obj, kp1,scene, kp2, good, None, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
show("matches",vis)

src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
H,mask = cv2.findHomography(src, dst, cv2.RANSAC,5.0)
print(H)

vis = cv2.drawMatches(obj,kp1,scene,kp2,good, None, matchesMask = mask.ravel().tolist(), flags =cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
show("inliners",vis)

h,w = obj.shape

corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
outline = cv2.perspectiveTransform(corners,H)

result = cv2.cvtColor(scene,cv2.COLOR_GRAY2BGR)
cv2.polylines(result ,[np.int32(outline)],True,(0,255,0),3)

show("found it :",result)