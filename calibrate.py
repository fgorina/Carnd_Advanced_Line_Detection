import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2
import pickle

### Generates calibration data and perspective matrix
# and stores the data in wide_dist_pickle.p


calibration_image_dir = "./images/camera_cal"
load_cal = True # Si false recalibrates came

if not load_cal:
    # prepare object points
    nx = 9#TODO: enter the number of inside corners in x
    ny = 6#TODO: enter the number of inside corners in y

    objp = np.zeros((ny*nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.


    ## Make a list of calibration images
    images = glob.glob('./images/camera_cal/calibration*.jpg')
    #images = glob.glob('./images/iphone/IMG_*.JPG')

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
            #write_name = 'corners_found'+str(idx)+'.jpg'
            #cv2.imwrite(write_name, img)
            cv2.imshow('img', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()


    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)


else:
    cal_data = pickle.load(open("./wide_dist_pickle.p", "rb"))
    mtx = cal_data["mtx"]
    dist = cal_data["dist"]

# Test undistortion on an image
img = plt.imread('./images/test_images/straight_lines2.jpg')
img_size = (img.shape[1], img.shape[0])


undist = cv2.undistort(img, mtx, dist, None, mtx)

plt.imsave('./images/test_images/undistorted2.jpg', undist)

# image1 correction
src = np.float32([[202., 720.], [582., 460.], [700., 460.], [1110., 720.]])
dst = np.float32([[202., 720.], [202., 100.], [1110., 100.], [1110., 720.]])

#image 2 correction
src = np.float32([[250., 700.], [598., 448.], [685., 448.], [1068., 700.]])
dst = np.float32([[250., 700.], [250., 00.], [1068., 00.], [1068., 700.]])

#image 2 correction bis
src = np.float32([[220., 720.], [580., 460.], [702., 460.], [1090., 720.]])
dst = np.float32([[220., 720.], [210., 130.], [1000., 130.], [1000., 720.]])



M = cv2.getPerspectiveTransform(src, dst)

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
dist_pickle["warp"] = M

pickle.dump(dist_pickle, open("./wide_dist_pickle.p", "wb"))

warped = cv2.warpPerspective(undist, M, img_size, flags=cv2.INTER_LINEAR)

plt.imsave('./images/test_images/warped2.jpg', warped)

#dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
# Visualize undistortion
f, axes = plt.subplots(2, 2, figsize=(20,10))

axes[0, 0].imshow(img)
axes[0, 0].set_title('Original Image', fontsize=30)
axes[0, 1].imshow(undist)
axes[0, 1].set_title('Undistorted Image', fontsize=30)
axes[1, 0].imshow(warped)
axes[1, 0].set_title('Warped Image', fontsize=30)
plt.show()