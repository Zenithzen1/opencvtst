import cv2
import numpy as np

def show(name, img) :
    cv2.imshow(name,img) #displays the graphical image
    cv2.waitKey(0)
    cv2.destroyAllWindows()

obj = cv2.imread("Degradation.jpeg",cv2.IMREAD_GRAYSCALE) #this returns a numpy array, pythons library for grids of numbers of the same type
scene = cv2.imread("scene.jpg",cv2.IMREAD_GRAYSCALE) #same as the above, in this case also the grid mainly represents brightnewssx
if obj is None or scene is None : 
    raise ValueError("Wrong Image file/No Image file. Check the folder")

print("object:", obj.shape, obj.dtype)
print("scene:",scene.shape,scene.dtype)
show("object",obj)

orb= cv2.ORB_create(2000) #ORB basically does 2 things, it first finds the keypoints, and then builds the descriptors around it and compares brightness of each pair of keypoints
kp1,des1 = orb.detectAndCompute(obj,None) #this function basically finds the keypoints and computes their descriptors in one call 
kp2, des2 = orb.detectAndCompute(scene,None)#sooo this thing basically comes in help when you need to match any 2 points

print("keypoints:",len(kp1),len(kp2))#a keypoint is basically a few pixels that are destinctive enough that you can spot them in another image
print("descriptors : ", des1.shape,des1.dtype) #compares the patches so a conclusion can be made about whether its similar enough or not
show("keypoints",cv2.drawKeypoints(obj,kp1,None, color = (0,255,0))) # points out all the keypoints

bf = cv2.BFMatcher(cv2.NORM_HAMMING) #BF stands for brute force, and norm hamming is basically for the amount of bits that differ
pairs =  bf.knnMatch(des1, des2, k=2 ) #we need to make comaprison so we need to know about the nearest neighbours
good =  [m for m, n in pairs if m.distance < 0.75 * n.distance] #
print("candidates: ",len(pairs),"after this ratio test:",len(good))

vis = cv2.drawMatches(obj, kp1,scene, kp2, good, None, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
show("matches",vis)

src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2) #query is for storing the matches 
dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2) #train index, altho its less training more just making use of data
H,mask = cv2.findHomography(src, dst, cv2.RANSAC,5.0)# 3x3 matrix that maps points on a flat surface in an image that basically sees what something is in one image to the other and also handles rotating and the sort
print(H)# It stands for RANdom SAmple Consensus, take any random 4 matches solve for H, count how many matches are within 5 pixels of H's prediction
#mask is basically a binary table, 0 if its an outlier, 1 if its an inlier for H

vis = cv2.drawMatches(obj,kp1,scene,kp2,good, None, matchesMask = mask.ravel().tolist(), flags =cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
show("inliners",vis)

h,w = obj.shape #height and width separately

corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2) #float because perspective transform requires it to be flaot
outline = cv2.perspectiveTransform(corners,H)

result = cv2.cvtColor(scene,cv2.COLOR_GRAY2BGR)
cv2.polylines(result ,[np.int32(outline)],True,(0,255,0),3) #chain of straight lines through a list of points, closes when it reaches a True that basically indicates that the object is complete and that full circle has been arrived.

show("found it :",result)