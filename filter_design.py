### Helps find correct parameters for filters

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import os
import math



def threshold_filter_yellow(h_channel, s_channel, v_channel, h_thresh=(0,30), s_thresh=(80,255), v_thresh=(70, 255)):

    h_channel = hsv[:,:,0]
    s_channel = hsv[:,:,1]
    v_channel = hsv[:,:,2]

    # Threshold saturation and intensity
    s_binary = np.zeros_like(s_channel)

    # Yellow Lines

    s_binary[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1]) &
             (v_channel >= v_thresh[0]) & (v_channel <= v_thresh[1]) &
             (s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1  # Compute average v_channel and set it at the middle. 2 Areas, > 400.


    return s_binary


### remove_dark will compute an average gray in the center (sure it is road)
#   and substitute dark zones with an average. This blurs dark zones
#   without modifying light ones.
#
#   img a single channel image
#
# returns
#
#   img with dark grays substituted
#
def remove_darkx(img):

    # lets get correct image data
    height = img.shape[0]
    width = img.shape[1]

    minx = int(width * 0.2)
    maxx = int(width * 0.8)

    minh = int(height * 0.7)
    maxh = height - 20 #remove desktop reflection
    avg_level = np.average(img[minh:maxh, minx:maxx])

    filtered = cv2.boxFilter(img,-1, (50,50))
    #filtered = cv2.medianBlur(img, 21)

    # OK now dark features are blurred
    # and now lower than averga go to 0-20%
    # and higer get expanded to 20-100%

    new_img = (np.copy(img) - avg_level)
    new_img = new_img / np.max(new_img) * 205 + 50

    new_img[(img < avg_level)] = 0


    filtered[(img >= avg_level)] = 0
    filtered = filtered / np.max(filtered) * 50

    new_img = np.add(new_img, filtered)
    return new_img.astype(np.uint8)


def remove_dark_slide(img_v, img_s):

    # lets get correct image data
    height = img_v.shape[0]
    width = img_v.shape[1]

    minx = int(width * 0.2)
    maxx = int(width * 0.8)

    minh = int(height * 0.7)
    maxh = height #remove desktop reflection
    avg_level = np.average(img_v[minh:maxh, minx:maxx])

    filtered_v = cv2.boxFilter(img_v,-1, (50,50))
    filtered_s = cv2.boxFilter(img_s,-1, (50,50))
    #filtered = cv2.medianBlur(img, 21)

    #new_img_v = np.copy(img_v)
    #new_img_v[(img_v < avg_level)] = 0
    new_img_v = (np.copy(img_v) - avg_level)
    new_img_v = new_img_v / np.max(new_img_v) * 205 + 50
    new_img_v[(img_v < avg_level)] = 0


    new_img_s = np.copy(img_s)
    new_img_s[(img_v < avg_level)] = 0

    filtered_v[(img_v >= avg_level)] = 0
    filtered_v = filtered_v/ np.max(filtered_v) * 50
    filtered_s[(img_v >= avg_level)] = 0

    new_img_v = np.add(new_img_v, filtered_v)
    new_img_s = np.add(new_img_s, filtered_s)


    return new_img_v, new_img_s

def remove_dark(img_v, img_s):

    height = img_v.shape[0]
    minh = int(height * 0.7)
    maxh = height - 20 #remove desktop reflection

    # Divide it into 2 horizontal slides

    n_slides = 4
    slide_height = int((maxh -minh) / n_slides)

    # Copy images

    new_img_v = np.copy(img_v)
    new_img_s = np.copy(img_s)

    for i in range(0,n_slides):

        s_minh = minh + (i * slide_height)
        s_maxh = s_minh + slide_height

        new_slide_v, new_slide_s = remove_dark_slide(img_v[s_minh:s_maxh, :], img_s[s_minh:s_maxh, :])

        new_img_v[s_minh:s_maxh, :] = new_slide_v
        new_img_s[s_minh:s_maxh, :] = new_slide_s

    return new_img_v, new_img_s


def color_filter(h_channel, s_channel, v_channel, orient='x', h_thresh=(100, 130), hx_thresh=(50,130), dilate=15, bins=7):     # Try to get gray. Forget the rest

    # lets get correct image data
    height = h_channel.shape[0]
    width = h_channel.shape[1]

    minx = int(width * 0.2)
    maxx = int(width * 0.8)

    minh = int(height * 0.7)
    maxh = height - 20 #remove desktop reflection


    # First try to get the correct hue values for the tartan
    h_histo = np.histogram(h_channel[minh:maxh, minx:maxx].flatten(), bins=bins)
    h_bin_max = np.argmax(h_histo[0])
    h_min = h_histo[1][h_bin_max]
    h_max = h_histo[1][h_bin_max + 1]


    # Threshold saturation and intensity
    h_channel_limited = np.copy(h_channel)
    h_channel_limited[(h_channel < h_min) | (h_channel > h_max)] = 0

    orient = 'x'

    if orient == 'x':
        sobel = cv2.Sobel(h_channel_limited, cv2.CV_64F, 1, 0, ksize=3) # Take the derivative in x
    else:
        sobel = cv2.Sobel(h_channel_limited, cv2.CV_64F, 0, 1, ksize=3)  # Take the derivative in x

    abs_sobel = np.absolute(sobel) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))




    # Threshold saturation and intensity
    s_binary = np.zeros_like(s_channel)
    s_binary[(scaled_sobel >= h_min) & (scaled_sobel <= h_max)] = 1

    kernel = np.ones((dilate, dilate), np.uint8)
    output = cv2.dilate(s_binary,kernel,iterations = 1)
    return output, scaled_sobel

def hue_filter(h_channel, s_channel, v_channel, orient='x', h_thresh=(100, 130),  dilate=15):     # Try to get gray. Forget the rest


    # Threshold saturation and intensity
    h_channel_limited = np.zeros_like(s_channel)
    h_channel_limited[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])] = 1


    kernel = np.ones((dilate, dilate), np.uint8)
    output = cv2.dilate(h_channel_limited, kernel, iterations = 1)
    return output


def abs_sobel_thresh(h_channel, s_channel, v_channel, orient='x', thresh=(12,255)):

    s_channel = s_channel.astype(np.float)
    v_channel = v_channel.astype(np.float)


    # Threshold saturation and intensity
    s_binary = np.zeros_like(s_channel)

    if orient == 'x':
        sobel = cv2.Sobel(v_channel, cv2.CV_64F, 1, 0, ksize=3) # Take the derivative in x
    else:
        sobel = cv2.Sobel(v_channel, cv2.CV_64F, 0, 1, ksize=3)  # Take the derivative in x

    abs_sobel = np.absolute(sobel) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    s_binary = np.zeros_like(s_channel, dtype=np.uint8)
    s_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return s_binary


def color_threshold(h_channel, s_channel, v_channel, s_thresh=(100,255), v_thresh=(50,255)):


    s_channel = s_channel.astype(np.float)
    v_channel = v_channel.astype(np.float)

    s_binary = np.zeros_like(s_channel, dtype=np.uint8)

    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])
        & (v_channel >= v_thresh[0]) & (v_channel <= v_thresh[1])] = 1

    return s_binary


def udacity_filter(h_channel, s_channel, v_channel, x_thresh=(12,255), y_thresh=(25,255), s_thresh=(75,255), v_thresh=(50,255)):

    grad_x = abs_sobel_thresh(h_channel, s_channel, v_channel, orient='x', thresh=x_thresh)
    grad_y = abs_sobel_thresh(h_channel, s_channel, v_channel, orient='y', thresh=y_thresh)
    c_binary = color_threshold(h_channel, s_channel, v_channel, s_thresh=s_thresh, v_thresh=v_thresh )

    mask = np.zeros_like(h_channel, dtype=np.uint8)

    mask[(((grad_x == 1) & (grad_y == 1)) | (c_binary == 1))] = 255

    return mask


def threshold_filter_white(h_channel, s_channel, v_channel, h_thresh=(0,255), s_thresh=(0,9), v_thresh=(100, 255)):

    h_channel = hsv[:,:,0]
    s_channel = hsv[:,:,1]
    v_channel = hsv[:,:,2]

    # Threshold saturation and intensity
    s_binary = np.zeros_like(s_channel)

    # Yellow Lines

    s_binary[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1]) &
             (v_channel >= v_thresh[0]) & (v_channel <= v_thresh[1]) &
             (s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1  # Compute average v_channel and set it at the middle. 2 Areas, > 400.


    return s_binary


def sobel_value_x_filter(h_channel, s_channel, v_channel, sobel_thresh=(12, 255), v_thresh=(50, 255), sw_thresh=(0,255)):

    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #v_channel = clahe.apply(v_channel)

    v_channel = cv2.equalizeHist(v_channel)
    s_channel = cv2.equalizeHist(s_channel)

    s_channel = s_channel.astype(np.float)
    v_channel = v_channel.astype(np.float)


    sobelx = cv2.Sobel(v_channel, cv2.CV_64F, 1, 0, ksize=3) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

      # Threshold saturation and intensity
    s_binary = np.zeros_like(s_channel)

    s_binary[(scaled_sobel >= sobel_thresh[0]) & (scaled_sobel <= sobel_thresh[1]) &
             (v_channel >= v_thresh[0]) & (v_channel <= v_thresh[1]) &
             (s_channel >= sw_thresh[0]) & (s_channel <= sw_thresh[1])] = 1




    return s_binary

def sobel_saturation_filter(h_channel, s_channel, v_channel, sobel_thresh=(15, 50), v_thresh=(70, 255), sy_thresh=(80,255)):

    s_channel = s_channel.astype(np.float)
    v_channel = v_channel.astype(np.float)


    sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0, ksize=3) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx) )


    # Threshold saturation and intensity
    s_binary = np.zeros_like(s_channel)

    s_binary[(scaled_sobel >= sobel_thresh[0]) & (scaled_sobel <= sobel_thresh[1]) &
             (v_channel >= v_thresh[0]) & (v_channel <= v_thresh[1]) &
             (s_channel >= sy_thresh[0]) & (s_channel <= sy_thresh[1])] = 1




    return s_binary

def sobel_value_y_filter(h_channel, s_channel, v_channel, sobel_thresh=(15, 50)):


    sobely = cv2.Sobel(v_channel, cv2.CV_64F, 0, 1, ksize=3) # Take the derivative in x
    abs_sobely = np.absolute(sobely) # Absolute x derivative to accentuate lines near from horizontal
    scaled_sobel = np.uint8(255*abs_sobely/np.max(abs_sobely))

    # Threshold saturation and intensity
#    s_binary = np.zeros_like(s_channel)

#    s_binary[(scaled_sobel >= sobel_thresh[0]) & (scaled_sobel <= sobel_thresh[1]) &
#             (v_channel >= v_thresh[0]) & (v_channel <= v_thresh[1]) &
#             (s_channel >= sw_thresh[0]) & (s_channel <= sw_thresh[1])] = 1

    return scaled_sobel

def gradient(h_channel, s_channel, v_channel, sobel_thresh=(15, 50)):


    sobelx = cv2.Sobel(v_channel, cv2.CV_64F, 1, 0, ksize=3) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal

    sobely = cv2.Sobel(v_channel, cv2.CV_64F, 0, 1, ksize=3) # Take the derivative in x
    abs_sobely = np.absolute(sobely) # Absolute x derivative to accentuate lines away from horizontal

    sobel_m = np.sqrt(sobelx * sobelx + sobely * sobely)
    sobel_d = np.arctan(sobely, sobelx) + math.pi/2

    s_binary = np.zeros_like(s_channel)

    s_binary[(sobel_d <= 0.2)] = 1


    return s_binary

def h_gradient(h_channel, s_channel, v_channel, sobel_thresh=(15, 50)):

    sobelx = cv2.Sobel(h_channel, cv2.CV_64F, 1, 0, ksize=3) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal

    sobely = cv2.Sobel(h_channel, cv2.CV_64F, 0, 1, ksize=3) # Take the derivative in x
    abs_sobely = np.absolute(sobely) # Absolute x derivative to accentuate lines away from horizontal

    sobel_m = np.sqrt(sobelx * sobelx + sobely * sobely)
    sobel_d = np.arctan(sobely, sobelx)

    s_binary = np.zeros_like(h_channel)

    s_binary[(abs(sobel_d) < 0.8)] = 1


    return s_binary


def onclick(event):
    print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          (event.button, event.x, event.y, event.xdata, event.ydata))


### Main program. Loads images, processes and shows different stages
## First show different components to analyze the images

c_id = -1

cal_data = pickle.load(open("./wide_dist_pickle.p", "rb"))
mtx = cal_data["mtx"]
dist = cal_data["dist"]

# Aquests valors semblen OK per video 1 and 3

def perspective(f=1.3245, h=460, size=(1280, 720), shrink=0.0, xmpp=0.004):
    l = size[0] * 0.2
    r = size[0] * 0.8
    l1 = l + size[0] * shrink
    r1 = r - size[0] * shrink
    b = size[1]
    src = np.float32([[l, b], [l + (b - h) * f, h], [r - (b - h) * f, h], [r, b]])
    dst = np.float32([[l1, b], [l1, 0.], [r1, 0.], [r1, b]])

    # Compute new xm_per_pix

    new_xmpp = xmpp / (r1 - l1) * (r - l)
    print(f)
    print(src)
    print(dst)
    M = cv2.getPerspectiveTransform(src, dst)
    ret, Minv = cv2.invert(M)

    return M, Minv, new_xmpp


def super_filter(h_channel, s_channel, v_channel, tipus='ud', prefilter=True):

    if tipus == 'ud':

        mask = udacity_filter(h_channel, s_channel, v_channel, x_thresh=(50, 255), y_thresh=(25, 255), s_thresh=(120, 255), v_thresh=(150, 255))

    else:
        sobel_v = sobel_value_x_filter(h_channel, s_channel, v_channel, sobel_thresh=(47, 255), v_thresh=(25, 255), sw_thresh=(0,255))
        sobel_s = sobel_saturation_filter(h_channel, s_channel, v_channel, sobel_thresh=(15, 50), v_thresh=(70, 255), sy_thresh=(80,255))
        mask = np.zeros_like(sobel_v)
        mask[((sobel_v == 1) | (sobel_s == 1))] = 255

    if prefilter:
        tartan, dummy = color_filter(h_channel, s_channel, v_channel, orient='x', dilate=20)
        combi = np.zeros_like(tartan)
        combi[((tartan == 1) & (mask == 255))] = 1
    else:
        combi = mask

    return combi

# Aquests valors semblen OK per video 2
#src = np.float32([[220., 720.], [580., 460.], [702., 460.], [1090., 720.]])
#dst = np.float32([[320., 720.], [0., -100.], [1190., -100.], [1000., 720.]])


#M = cv2.getPerspectiveTransform(src, dst)
M, Minv, xp= perspective(f=1.263, h=550, shrink=0.15)

#M, Minv = perspective(f=1.393, h=475, shrink=0.1) # For video 2
#M, Minv = perspective(f=1.263, h=500, shrink=0.15) # For video 3

image_folder = "./images/log_images2"


for file in os.listdir(image_folder):

    if file.endswith("128.jpg"):

        image = cv2.imread(image_folder + "/" + file)
        image = cv2.undistort(image, mtx, dist, None, mtx)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h_channel = hsv[:, :, 0]
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]


        eqs = cv2.equalizeHist(s_channel)
        eqv = cv2.equalizeHist(v_channel)

        eqs = cv2.GaussianBlur(eqs, (7, 7), 0., 0.)
        eqv = cv2.GaussianBlur(eqv, (7, 7), 0., 0.)

        filtered_eqv, filtered_eqs = remove_dark(eqv, eqs)
        eqv = filtered_eqv
        #eqs = filtered_eqs

        r_ud_n = super_filter(h_channel, eqs, eqv, tipus='ud', prefilter=False)
        r_ud_y = super_filter(h_channel, eqs, eqv, tipus='ud', prefilter=True)
        r_my_n = super_filter(h_channel, eqs, eqv, tipus='my', prefilter=False)
        r_my_y = super_filter(h_channel, eqs, eqv, tipus='my', prefilter=True)


        ud_mask = cv2.warpPerspective(r_ud_y, M, (r_ud_y.shape[1], r_ud_y.shape[0]), flags=cv2.INTER_LINEAR)
        my_mask = cv2.warpPerspective(r_my_y, M, (r_my_y.shape[1], r_my_y.shape[0]), flags=cv2.INTER_LINEAR)

        # color selection es el meu algoritne, mask el udacity
        #

        # Now apply a perspective transformation


        # Compute histograms

        b = 7
        dilate = 20

        kernel = np.ones((dilate, dilate), np.uint8)

        h_histo = np.histogram(h_channel[430:700,:].flatten(), bins=b)
        h_bin_max = np.argmax(h_histo[0])
        h_min = h_histo[1][h_bin_max]
        h_max = h_histo[1][h_bin_max+1]
        h_img = np.zeros_like(h_channel)
        h_img[((h_channel >= h_min) & (h_channel <= h_max))] = 255
        h_img = cv2.dilate(h_img, kernel, iterations=1)



        s_histo = np.histogram(eqs[430:700,].flatten(), bins=b)
        s_bin_max = np.argmax(s_histo[0])
        s_min = s_histo[1][s_bin_max]
        s_max = s_histo[1][s_bin_max + 1]
        s_img = np.zeros_like(s_channel)
        s_img[((s_channel >= s_min) & (s_channel <= s_max))] = 255
        s_img = cv2.dilate(s_img, kernel, iterations=1)

        empty = np.zeros_like(s_channel)
        newdata = np.stack((empty , empty, h_img), axis=2)

        grad = h_gradient(h_channel, eqs, eqv)


        tartan_x, sobel = color_filter(h_channel, eqs, eqv, orient='x', h_thresh=(100, 130), hx_thresh=(50,130), dilate=15)
        tartan_y = color_filter(h_channel, eqs, eqv, orient='y', h_thresh=(100, 130), hx_thresh=(50,130), dilate=15)


        um =  np.zeros_like(s_channel)
        um[((sobel == 1) & (grad == 1))] = 1

        f, axes = plt.subplots(4, 2, figsize=(24, 9))
        axes[0,0].hist(h_channel[430:700,:].flatten(), bins=b)
        axes[0,1].hist(s_channel[430:700,:].flatten(), bins=b)
        #axes[1,0].imshow(newdata)
        #axes[1,1].imshow(h_channel, cmap="gray")
        axes[1,0].imshow(r_ud_n, cmap="gray")
        axes[1,1].imshow(r_my_n, cmap="gray")
        #axes[2,0].imshow(s_img, cmap="gray")
        #axes[2,1].imshow(s_channel, cmap="gray")
        axes[2, 0].imshow(r_ud_y, cmap="gray")
        axes[2, 1].imshow(r_my_y, cmap="gray")
        #axes[3, 0].imshow(grad, cmap="gray")
        #axes[3, 1].imshow(sobel, cmap="gray")
        axes[3, 0].imshow(eqv, cmap="gray")
        axes[3, 1].imshow(filtered_eqv, cmap="gray")
        axes[0, 0].set_title(file)
        axes[0, 1].set_title("Saturation Histogram")
        axes[1, 0].set_title("UD_N")
        axes[1, 1].set_title("MY_N")
        axes[2, 0].set_title("UD_Y")
        axes[2, 1].set_title("MY_Y")

        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.1)

        plt.show()

