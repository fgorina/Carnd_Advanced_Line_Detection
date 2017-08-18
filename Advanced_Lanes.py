import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import os
import math
import copy
import datetime


'''Conscient Filter is a new approach to the filtering

    It not oly filters the images but maintains a "world belief"
    manages its "memory" and perfectly separates between the "belief", the "measures"
    and the "memory"

    Units are changed to made things easier. y = 0 is the bottom of the image.

'''

### UTILITY FUNCTIONS

### gaussian
#
#   gaussian function with mean mu and stdev sig
#
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * sig))

def sigma2(p0):
    return -np.log(p0)/2

### perspective generates a Perspective transformation given f
#
#   f is a variable that tries to get the focal of the camera
#   h is the top point. Its points will be mapped to y = 0.0
#       Points over h will not be considered for fitting.
#   size is the size of the std image
#   shrink is a proportion of the image width to be used to
#   reduce width betwewn lanes in pixels so curves fit in the window
#   If shrink != 0 then the meters for x pixels should be modified acordingly
#

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

### first_not_none returs first element of list which is not None
#

def first_not_none(l):
    for item in l:
        if item is not None:
            return item

    return None

### Fit is a class that represents the results of a fit.
#
#   Its structure depends of the equation that will be used (subclass).
#   Here is a second order polygon
#
class Fit:

    def __init__(self):

        #coeficients is an array with A, B, C of the equation
        self.coeficients = None
        self.residuals = None
        self.world_coeficients = None

        # Originalx, y values
        self.x_values = None
        self.y_values = None

        # xm, ym from the original data
        self.xm = 1.0
        self.ym = 1.0

    ### new_fit generates a new Fit from two lists of corresponding x, y points

    @classmethod
    def new_fit_from_points(self_class, fit_points_x, fit_points_y, xm, ym):

        fit = Fit()
        fit.x_values = fit_points_x
        fit.y_values = fit_points_y
        fit.xm = xm
        fit.ym = ym
        fit.compute_fit()

        return fit

    ### compute_world_coeficients computes coeficients in world coordinates from local ones

    def compute_world_coeficients(self):
        self.world_coeficients = np.copy(self.coeficients)
        self.world_coeficients[0] = self.coeficients[0] * self.xm / (self.ym * self.ym)
        self.world_coeficients[1] = self.coeficients[1] * self.xm / self.ym
        self.world_coeficients[2] = self.coeficients[2] * self.xm

    ### Computes the fit from the x_values, y_values

    def compute_my_residuals(self):

        xc = self.value(self.y_values) - self.x_values
        xc2 = xc * xc
        xc2m = np.average(xc2)
        xcm = np.average(np.abs(xc))
        xcm2 = xcm * xcm
        sd = np.std(xc)
        sd2 = sd * sd

        # An alternative is to adjust a linear fit between xc and y

        A = np.vstack([self.y_values, np.ones(len(self.y_values))]).T

        error_fit = np.linalg.lstsq(A, np.abs(xc))

        e_0 = error_fit[0][1]
        e_m = error_fit[0][0]



        return xc2m, sd

    def compute_sd2(self):
        xc = self.value(self.y_values) - self.x_values
        sd = np.std(xc)

        return sd*sd

    def compute_fit(self):

        aux = np.polyfit(self.y_values, self.x_values, 2, full=True)
        self.coeficients = aux[0]
        if len(aux[1]) > 0:
            self.residuals = aux[1][0] / len(self.x_values)

        else:
            self.residuals = 500  # Must work, usually too few points


        #my_residuals, mysd = self.compute_my_residuals()
        self.compute_world_coeficients()

    # Returns the x value from the y value

    def value(self, y):
        v = self.coeficients[0]*y*y + self.coeficients[1]*y + self.coeficients[2]
        return v

    # Returns derivative at y position

    def prime_value(self, y):
        return 2.0 * self.coeficients[0]*y + self.coeficients[1]

    # Returns the x value from the y value in world coordinates. Be carful y in world coordinates also

    def world_value(self, y):
        return self.world_coeficients[0] * y * y + self.world_coeficients[1] * y + self.world_coeficients[2]

    # Returns the x' value from the y value in world coordinates. Be careful y in world coordinates also

    def world_prime_value(self, y):
        return 2.0 * self.world_coeficients[0] * y + self.world_coeficients[1]

    # Returns radius of curvature at y

    def radius(self, y):
        curverad = ((1 + (2 * self.coeficients[0] *y + self.coeficients[1]) ** 2) ** 1.5) / np.absolute(2 * self.coeficients[0])
        return curverad

    # Returns radius of curvature at y in world values. Be careful y in world coordinates also

    def world_radius(self, y):
        curverad = ((1 + (2 * self.world_coeficients[0] * y + self.world_coeficients[1]) ** 2) ** 1.5) / np.absolute(
            2 * self.world_coeficients[0])

        return curverad

    # move fit generates a new fit when the observer moves d along y direction and the corresponding y value
    #
    #   sigma es l'error en d. maxy es la maxima y a on considerar els errors
    #   Es una modificacio directa del fit per lo que no han d'haver-hi valors
    #   a x_values, y_values
    #

    def move(self, d, sigma, maxy):

        new_fit = Fit()
        new_fit.xm = self.xm
        new_fit.ym = self.ym

        new_fit.coeficients = copy.copy(self.coeficients)
        new_fit.coeficients[1] = (2 * self.coeficients[0] * d) + self.coeficients[1]

        # Sigma is added sigmas byt x sigma must be computed fromsigma in v
        derror = (2 * self.coeficients[0] * maxy + self.coeficients[1])*sigma

        new_fit.residuals = np.power(np.sqrt(self.residuals) + abs(derror), 2)

        new_fit.compute_world_coeficients()

        return new_fit

    # redo_fit_moved fit generates a new fit but as if we move the points
    #
    #   sigma es l'error en d. maxy es la maxima y a on considerar els errors
    #   x_values, y_values are moved and the fit recomputed.
    #   Then the sigmas composed
    #

    def redo_fit_moved(self, d, sigma, maxy):

        xm = self.value(d)

        new_fit = Fit()
        new_fit.x_values = self.x_values - xm
        new_fit.y_values = self.y_values - d
        new_fit.xm = self.xm
        new_fit.ym = self.ym

        new_fit.compute_fit()

        # Sigma is added sigmas byt x sigma must be computed from sigma in v

        new_fit.residuals = new_fit.residuals + (self.coeficients[0] * maxy + self.coeficients[1])*d

        return new_fit


    # compose_fit computes a new fit average of the fits according its sigmas :
    #
    #   new_fit = (fit1 * fit2.sigma + fit2 * fit.sigma ) / (sigma1 + sigma2
    #   new sigma = 1/((1/sigma1)+(1/sigma2))
    #   X_values, y_values are lost

    def compose_fit(self, fit):

        new_fit = Fit()

        new_fit.xm = self.xm
        new_fit.ym = self.ym

        new_fit.coeficients = (self.coeficients * fit.residuals + fit.coeficients * self.residuals)/(self.residuals + fit.residuals)
        new_fit.residuals = 1.0/((1.0/self.residuals) + (1.0/fit.residuals))
        new_fit.compute_world_coeficients()
        return new_fit

        # compose_fit computes a new fit average of the fits according its sigmas :
        #
        #   new_fit = (fit1 * fit2.sigma + fit2 * fit.sigma ) / (sigma1 + sigma2
        #   new sigma = 1/((1/sigma1)+(1/sigma2))
        #   X_values, y_values are lost

    def weigthed_fit(self, w1,  fit, w2):

        new_fit = Fit()

        new_fit.coeficients = (self.coeficients * w1 + fit.coeficients * w2) / (w1 + w2)
        new_fit.residuals = 1.0 / ((1.0 / self.residuals) + (1.0 / fit.residuals))
        new_fit.compute_world_coeficients()



    # unionfit computes a new fit from the union of the points.  :
    #
    #   All x, y values are added together
    #   The fit is recomputed
    #

    def union_fit(self, fit):

        new_fit = Fit()

        new_fit = Fit()
        new_fit.x_values = np.concatenate((self.x_values, fit.x_values))
        new_fit.y_values = np.concatenate((self.y_values, fit.y_values))
        new_fit.xm = self.xm
        new_fit.ym = self.ym

        new_fit.compute_fit()

        return new_fit

    # average_fit computes the average fit of other 2 ones.
    #
    #   It is just the average without weights.
    #   sigma2 is the sum of the sigmas. perhaps only the average?
    #
    def average_fit(self, fit):

        new_fit = Fit()
        new_fit.xm = self.xm
        new_fit.ym = self.ym
        new_fit.coeficients = (self.coeficients + fit.coeficients)/2.0
        new_fit.residuals = self.residuals + fit.residuals
        new_fit.compute_world_coeficients()

        return new_fit


### Side_Measure is the results of a measure of a single lane, either left or right
#
#   It includes the fits and the data used to get them
#

class Lane_Measure():
    def __init__(self):

        # Side of the measure, (left or right)
        self.side = None

        # Fit   (Result of the fit)
        self.fit = None

        # Sigma2 (average of fit residues)
        self.sigma2 = 0.0

        # selection_points are the area over which image points are selected for the fit
        # Is in the form of a mask of the size of the image used.
        # Be careful because these points are not  in our reversed coordinates so they are
        # compatible with the images
        #
        self.selection_points = None

        # filter is a string identifying filter and possible parameter values
        self.filter = None

        # method is the method used to get the fit points. either 'windows' or 'old_fit'
        self.method = None

        # old_fit is a Fit structure of the fit used if method is old_fit
        self.old_fit = None

        # radius is a convenience value with the of curvature radius at y = 0

        self.radius = None

    @classmethod
    def new_lane_measure_from_data(self_class, x_points, y_points,  selection_points, filter_x=None, side='left', method='windows', old_fit=None, xm=1.0, ym=1.0):

        lane_measure = Lane_Measure()

        lane_measure.side = side
        lane_measure.fit = None
        lane_measure.sigma2 = 0.0
        lane_measure.selection_points = selection_points
        lane_measure.filter = filter_x
        lane_measure.method = method
        lane_measure.old_fit = old_fit
        lane_measure.radius = None
        lane_measure.fit = Fit.new_fit_from_points(x_points, y_points, xm, ym)
        lane_measure.radius = lane_measure.fit.world_radius(0.0)
        lane_measure.sigma2 = lane_measure.fit.residuals

        return lane_measure

    # Advance just updates its stane, doesn't generate a new one but creates a new fit
    def advance(self, d, sigma2):
        ymax = self.get_shape()[0]  # Just for computing new errors
        self.old_fit = self.fit     # Just store old fit!!!
        self.fit = self.fit.move(d, sigma2, ymax)   # Create a new one moved

        self.radius = self.fit.world_radius(0.0)    # Compute new radius
        self.sigma2 = self.fit.residuals            # Set new residuals


    ### Returns the x value for an y
    #
    def get_x(self, y):
        if self.fit is None:
            return None
        else:
            return self.fit.value(y)

    ### Returns the x value for an y_w in world coordinates
    #
    def get_x_w(self, y):
        if self.fit is None:
            return None
        else:
            return self.fit.world_value(y)

    ### Returns the x' value for an y
    #
    def get_x_prime(self, y):
        if self.fit is None:
            return None
        else:
            return self.fit.prime_value(y)

    ### Returns the x' value for an y in world coordinates
    #

    def get_x_prime_w(self, y):
        if self.fit is None:
            return None
        else:
            return self.fit.world_prime_value(y)

    ### Returns x position at y = 0 (my car)

    def get_base_x(self):
         return self.get_x(0.0)

    def get_base_x_w(self):
         return self.get_x_w(0.0)

    ### Reuturns the points of the original data

    def get_shape(self):
        if self.selection_points is None:
            return (720, 1280)
        else:
            return self.selection_points.shape


     # Computes a probability that another fit may be compatible given their sigmas

    def p_measure(self, m2, offset):

        sig2 = np.power(math.sqrt(self.sigma2) + math.sqrt(m2.sigma2), 2)       # Sum of sigmas
        maxy = self.get_shape()[0]

        yvals = np.linspace(0, maxy - 1, maxy)
        xv1 = self.get_x(yvals) + offset
        xv2 = m2.get_x(yvals)

        p = gaussian(xv2, xv1, sig2)

        return np.average(p)

    ## p_dist returns a number between 0 and 1 where 0 means all bins of a 5 bin histogram are full
    #   and 1 means are empty

    def p_dist(self):

        ymax = self.get_shape()[0]
        histo = np.histogram(self.fit.y_values, bins=5, range=(0, ymax))

        p = 0.0
        for v in histo[0]:
            if v < 100:
                p += 0.2

        return p

    ## p_sigma is a value that if sigma = 0 its value is 1 and if sigma is very big its value is 0
    #   it is scaled by a 400 factor that is the "average width" of a lane

    def p_sigma(self):

        return np.exp(-np.sqrt(self.sigma2)/400)

    ## p_score is a value with 0 means a perfect score and 1 means a horrible one
    #
    #   It depends that my distribution of points is right and I have a low sigma

    def p_score(self):

        p_d = 1 - self.p_dist()
        p_s = self.p_sigma()

        #print("    Scoring Lane {} p_dist {:0.2f} p_sigma {:0.2f}".format(self.filter, p_d, p_s))
        return p_d * p_s

    # Computes a probability that another fit may be compatible given their sigmas
    #   But forgets segond sigma so bad fits are not given a boost

    def p_strict_measure(self, m2):


        maxy = self.get_shape()[0]

        yvals = np.linspace(0, maxy - 1, maxy)
        xv1 = self.get_x(yvals)
        xv2 = m2.get_x(yvals)

        p = gaussian(xv2, xv1, 2*self.sigma2)

        return np.average(p)

    # Computes a

    # Returns a new measure that is the average of itself and other_measure
    #   Sigma2 are added as there is no composition, just add and divide by 2

    def average(self, other_measure):

        new_lane_measure = Lane_Measure()

        new_lane_measure.filter = "x: " +self.filter + ", "+ other_measure.filter
        new_lane_measure.method = "x: " +self.method + ", " + other_measure.filter
        new_lane_measure.old_fit = [self.old_fit, other_measure.old_fit]

        new_lane_measure.side = "center"
        new_lane_measure.selection_points = np.concatenate((self.selection_points, other_measure.selection_points))

        new_lane_measure.fit = self.fit.average_fit(other_measure.fit)

        new_lane_measure.sigma2 = new_lane_measure.fit.residuals
        new_lane_measure.radius = new_lane_measure.fit.world_radius(0.0)

        return new_lane_measure

### Measure represents the result of a measure.
#
#   Each filter process generates a new measure
#   To create a Masure compute the 2 side_measures and use them to create the Measure
#

class Measure:
    def __init__(self, left, right, image):

        # left data is left Side_Measure
        self.left_data = left

        # right data is right Side_Measure
        self.right_data = right

        # min max and average lane width

        self.min_lane = None
        self.max_lane = None
        self.y_min_lane = None
        self.y_max_lane = None
        self.average_lane = None

        # warped image used to get the data

        self.warped_image = image

        self.compute_data()



    def compute_data(self):

        if self.left_data is None or self.right_data is None:
            return

        ymax = self.left_data.get_shape()[0]    # Get the number of points

        ploty = np.linspace(0, ymax - 1, ymax)
        left_fitx = self.left_data.get_x(ploty)
        right_fitx = self.right_data.get_x(ploty)
        delta = right_fitx - left_fitx

        self.min_lane = min(delta)
        self.max_lane = max(delta)
        self.avg_lane = np.average(delta)

        self.y_min_lane = np.argmin(delta, 0)
        self.y_max_lane = np.argmax(delta, 0)


    def lane_width(self, y):

        if self.left_data is None or self.right_data is None:
            return

        return self.right_data.get_x(y) - self.left_data.get_x(y)


    def lane_width_w(self, y):
        if self.left_data is None or self.right_data is None:
            return

        left_v = self.left_data.get_x_w(y)
        right_v =  self.right_data.get_x_w(y)

        v = right_v - left_v
        return v

    def lane_center(self, y):

        if self.left_data is None or self.right_data is None:
            return

        return (self.right_data.get_x(y) + self.left_data.get_x(y))/2.0

    def lane_center_w(self, y):
        if self.left_data is None or self.right_data is None:
            return

        return (self.right_data.get_x_w(y) + self.left_data.get_x_w(y))/2.0

    ## divergence computes difference in width between y0 and y1 as
    #
    #   divergence = (width(y1) - width(y0))/width(y0)

    def divergence(self, y0, y1):

        return (self.lane_width(y1) - self.lane_width(y0))/self.lane_width(y0)

    ## p_ok is the result of some calculation thats gives it a POK
    #
    #   Now is the p entre els dos fits moved offset
    #

    def p_measure(self):

        if self.left_data.sigma2 is None:
            print("Left")

        if self.right_data.sigma2 is None:
            print("Left")


        #print(" sigma left {:0.2f} right {:0.2f}".format(self.left_data.sigma2, self.right_data.sigma2))
        sig2 = np.power(math.sqrt(self.left_data.sigma2) + math.sqrt(self.right_data.sigma2), 2)

        offset = self.right_data.get_x(0)-self.left_data.get_x(0)

        maxy = self.left_data.get_shape()[0]
        yvals = np.linspace(0, maxy - 1, maxy)
        xv1 = self.left_data.get_x(yvals) + offset
        xv2 = self.right_data.get_x(yvals)

        p = gaussian(xv2, xv1, sig2)

        pm = np.average(p)
        return pm

    ### p_score is 0 for a perfect one and 1 for a horrible one
    #

    def p_score(self):

        p_left = self.left_data.p_score()
        p_right = self.right_data.p_score()
        p_m = self.p_measure()

        p = p_left * p_right * p_m

        #print("Scoring {} : {:0.2f} Left {:0.2f} Right {:0.2f} Measure {:0.2f}".format(self.left_data.filter, p, p_left, p_right, p_m))

        return p

### Class Calibraton has some calibration data of my senses
#
#   Speed should not be here really
class Calibration:
    def __init__(self, mtx, dist, m, m_inv, xm, ym):

        # Camera matrix
        self.mtx = mtx

        # Distortion correction coficients
        self.dist = dist

        # Perspective matrix
        self.M = m

        # Inverse perspective Matrix
        self.Minv = m_inv

        # c meters per pixel
        self.xm = xm

        # y meters per pixel
        self.ym = ym

### Belief is my instanvt vision of the world
#   Of course is a result of perception and reconstruction
#   so everyhing is possible

class Belief(Measure):
    def __init__(self):

        # what frame is this Instant related?
        self.frame = 0

        # center_lane is computed from data and is our center_lane. It is a Lane_Measure
        self.center_lane = None

        # measures are the measures that take us from frame before to this one
        self.measures = None

        # probability of being ok
        self.prob = 0.5

    ## Compute center_lane computes center_lane from left_data and center_data
    #
    #
    def compute_center_lane(self):

        self.center_lane = self.left_data.average(self.right_data)



    ## Some data only available in beliefs and not in measures

    ## center_radius returns the radius of the center fit

    def center_radius(self):

        return self.center_lane.radius

    ## Position returns the offset from the center of the lane
    #  < 0 means I am at right, > 0 at left
    #
    def position(self):

        image_center = self.left_data.get_shape()[1] / 2.0
        lane_center = self.center_lane.get_x(0)

        lane_offset = lane_center - image_center  # < 0 means I am at right, > 0 at left

        return lane_offset

    ## position_w gives the same as position in world coordinates
    #
    def position_w(self):

        return self.position() * self.left_data.fit.xm

    ## Some functions to update a belief and contrast it against measures
    #

    ## predict will try to predict the future generating a new Belief
    #   advanced in the future
    #
    def predict(self, speed, sigma2, steps=1):

        new_belief = copy.deepcopy(self)

        new_belief.left_data.advance(speed, sigma2)
        new_belief.right_data.advance(speed, sigma2)
        new_belief.compute_data()
        new_belief.compute_center_lane()

        return new_belief




    ## compose adds a measure to the belief modifying its data
    #   evaluation is a measure of how good is the measure. usually a return from plausability
    #
    def compose(self, measure, evaluation):

        # Compute new coeficients proportional to probabilities

        self.left_data.fit = self.left_data.fit.compose_fit(measure.left_data.fit)
        self.right_data.fit = self.right_data.fit.compose_fit(measure.right_data.fit)
        self.left_data.radius = self.left_data.fit.world_radius(0.0)
        self.left_data.sigma2 = self.left_data.fit.residuals
        self.right_data.radius = self.right_data.fit.world_radius(0.0)
        self.right_data.sigma2 = self.right_data.fit.residuals
        self.warped_image = measure.warped_image


        self.measures = [measure]
        self.compute_data()
        self.compute_center_lane()

       # #print("sigma2 {:0.2f} {:0.2f}".format(self.left_data.sigma2, self.right_data.sigma2))

### Belief represents my belief about how is my world
#
#   First have some stable data about ME and about how is the world made
#   Second has my idea of

class World:
    def __init__(self, calibration, lane_widths=(3.0, 4.5), min_lane_curvature=200, max_base_jump=20, speed=24, speed_sigma2=14):

        # First some stable data

        # An instance of my calibration data
        self.calibration = calibration

        # Minimum lane width
        self.lane_widths = lane_widths

        # Minimum lane curvature
        self.minimum_lane_curvature = min_lane_curvature

        # Maximum base jump is maximum jump from one measure to next. in some way linked to speed
        self.max_base_jumo = max_base_jump

        # Speed is my perceived spped. Now is fixed
        self.speed = speed

        #speed_sigma2 is the error generated when I apply speed
        self.speed_sigma2 = speed_sigma2

        # Sliding window parameters

        self.window_width = 50
        self.window_height = 80
        self.window_margin = 100
        self.margin = 100
        self.min_points = 5

        # Triggers a world.reset
        self.max_skipped = 10
        # Triggers a sliding windows next measure
        self.use_windows = 3

        # Actual frame

        self.frame = 0
        # History is a list of beliefs. We prepend (push) so history[0] is our actual belief already verified

        # skipped frames
        self.skipped = 0

        self.history = []

        # working_belief is a prediction or work in progress
        #
        #   Once validated is pushed to history

        self.working_belief = None

        # debug list
        #
        #   It is a list with pairs of time, string, image
        #   where to store data when executing the program for debugging
        #
        #   For example intermediate images

        self.debug_list = []

        # width of lane and error
        #
        self.lane_width = 425
        self.lane_sigma2 = 250

        # llista de radius

        self.radius_list = []
        self.n_avg_radius = 5

        # maximum distance pposition from cneter lane in m

        self.max_position = 1.0



    # Resets history

    def reset(self):
        self.working_belief = None
        self.history = []
        self.frame = 0
        self.debug_list = []
        self.radius_list = []



    # Advance mades a prediction in time

    def advance(self, to_frame):

        self.frame = to_frame

        if self.working_belief is not None and world.skipped <= 5:
            self.working_belief = self.working_belief.predict(self.speed, self.speed_sigma2, steps=1) #to_frame-frame)
            self.working_belief.frame = to_frame

    ## plausability is a measure as how plausably si a measure from a belief
    #
    #   returns p_total, p_left, p_right as probabilities and info as a reasoned string []
    #
    def plausability(self, measure):
        p_m = measure.p_score()
        p_left = self.left_data.p_measure(measure.left_data, 0)  # prova sense strict
        p_right = self.right_data.p_measure(measure.right_data, 0)
        p_total = p_left * p_right * p_m

        return p_total, p_left, p_right, p_m, ["I believe everything"]

    ## select_best selects a best measure from a list and returns the measure and plausability
    #
    #   We made the cross product of all left and right measures and compute
    #   the p_score of each pair
    #
    #   Get the one with best p_score
    #


    def sanity_check(self, left, right, p_left, p_right, p_m):

        # First check that lane with is reasonable


        wl = (right.get_x(0) - left.get_x(0))*self.calibration.xm
        wl_top= (right.get_x(720) - left.get_x(720))*self.calibration.xm


        if wl < self.lane_widths[0] or wl > self.lane_widths[1]:
            return False

        if self.working_belief is not None:
            position = self.working_belief.position_w()

            if abs(position) > self.max_position:
                print("LPosition error {} - {}".format(position))
                return False

            if abs(wl - self.lane_width*self.calibration.xm) > 2*max(0.30, self.lane_sigma2 * self.calibration.xm):
                print("Lane Width error {} ".format(wl))
                return False

            if abs(wl_top - wl) > 2:
                print("Top Lane Width error {} - {}".format(wl, wl_top))
                return False

            if p_m < .5:
                print("p error {} ".format(p_m))
                return False
        # May add things as curvature (2A) or prime(0) but have no idea of correct values



        return True


    def select_best(self, measures):

        # Compute an array of p_socre for each measure, left and right

        scores = []  # Measure, left p_score, right p_score

        p_selected = 0.0
        l_selected = None
        r_selected = None
        p_l_selected = 0
        p_r_selected = 0
        p_m_selected = 0

        img_selected = None

        for m in measures:
            if m is not None and m.left_data is not None and m.right_data is not None:
                scores.append([m, m.left_data.p_score(), m.right_data.p_score()])

        for m0 in scores:
            for m1 in scores:
                left = m0[0].left_data
                right = m1[0].right_data
                p_left = m0[1]
                p_right = m1[2]
                offset = right.get_x(0) - left.get_x(0)
                p_m = left.p_measure(right, offset)

                p_t = p_left * p_right * p_m

                if p_t > p_selected and self.sanity_check(left, right, p_left, p_right, p_m):
                    p_selected = p_t
                    l_selected = left
                    r_selected = right
                    p_l_selected = p_left
                    p_r_selected = p_right
                    p_m_selected = p_m
                    if p_left > p_right:
                        img_selected = m0[0].warped_image
                    else:
                        img_selected = m1[0].warped_image

        if l_selected is None or r_selected is None:
            new_measure = None
        else:
            new_measure = Measure(l_selected, r_selected, img_selected)

        return new_measure, (p_selected, p_l_selected, p_r_selected, p_m_selected)

    ### That is the main function to be called
    #
    #   It receives a set of measures and updates its believe and pushes the belief to history
    #   returns the update belief
    #
    #   The working belief has already been advances.
    #



    def update(self, measures):

        # we select best measure

        best, evaluation = self.select_best(measures)

        if best is None : # No good measure
            self.skipped += 1

            return self.working_belief

        # First case, we know nothing about the world
        # Try the best measure we have and let it a go.
        #
        # Probably we need some aditional sanity check just in case

        if self.skipped >= self.max_skipped:
            world.reset()

        if self.working_belief is None:

            self.working_belief = Belief()
            self.working_belief.frame = self.frame
            self.working_belief.left_data = best.left_data
            self.working_belief.right_data = best.right_data
            self.working_belief.warped_image = best.warped_image
            self.working_belief.compute_data()
            self.working_belief.compute_center_lane()
            self.working_belief.measures = [best]

        else:
            # All seems correct, update our position in the world
            self.working_belief.compose(best, evaluation)



        self.skipped = 0
        self.history[:0] = [self.working_belief]

        # Now we must compute new width lane

        self.compute_lane_width()
        # TODO modify to store it in a CVS log. Much more useful
        #
        #   Also add info about curvatures, probabilities
        #
        p = self.working_belief.p_measure()
        print("Frame {} updated {:0.2f} Sigma left {:0.2f} Sigma right {:0.2f} Divergence {:0.2f} Left a {} right a {} ".format(self.frame, p,
                                                self.working_belief.left_data.sigma2,
                                                self.working_belief.right_data.sigma2,
                                                self.working_belief.divergence(0,720) * 100,
                                                self.working_belief.left_data.fit.coeficients[0],
                                                self.working_belief.right_data.fit.coeficients[0]))

        return self.working_belief


    def compute_lane_width(self):

        new_lane_width = self.working_belief.lane_width(0)
        new_lane_sigma2 = (self.working_belief.left_data.sigma2 + self.working_belief.right_data.sigma2) / 2

        if self.lane_width is None:
            self.lane_width = new_lane_width
            self.lane_sigma2 = new_lane_sigma2
        else:
            self.lane_width = (new_lane_width * self.lane_sigma2 + self.lane_width * new_lane_sigma2) / (
            self.lane_sigma2 + new_lane_sigma2)
            self.lane_sigma2 = 1 / (1 / self.lane_sigma2 + 1 / new_lane_sigma2)


        # Now add radius to radius_list

        self.radius_list.append(self.working_belief.center_radius())
        while len(self.radius_list) > self.n_avg_radius:
            self.radius_list.pop(0)

    def get_avg_radius(self):

        if len(self.radius_list) > 0:
            return np.average(self.radius_list)

        else:
            return -1

    def add_debug(self, text, image):

        l = [datetime.datetime.now(), text, image]
        self.debug_list.append(l)

#
### FILTER FUNCTIONS

### FILTERS - Different filters to use

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
def remove_dark_slide(img_v, img_s):

    # lets get correct image data
    height = img_v.shape[0]
    width = img_v.shape[1]

    minx = int(width * 0.2)
    maxx = int(width * 0.8)

    minh = 0
    maxh = height #remove desktop reflection
    avg_level = np.average(img_v[minh:maxh, minx:maxx])

    filtered_v = cv2.boxFilter(img_v,-1, (51,51))
    filtered_s = cv2.boxFilter(img_s,-1, (51,51))
    #filtered = cv2.medianBlur(img, 21)

    #new_img_v = np.copy(img_v)
    #new_img_v[(img_v < avg_level)] = 0
    new_img_v = (np.copy(img_v) - avg_level)
    new_img_v = new_img_v / np.max(new_img_v) * 205 + 50
    new_img_v[(img_v < avg_level)] = 0


    new_img_s = np.copy(img_s)
    new_img_s[(img_v < avg_level)] = 0

    filtered_v[(img_v >= avg_level)] = 0
    filtered_v = filtered_v / np.max(filtered_v) * 50
    filtered_s[(img_v >= avg_level)] = 0

    new_img_v = np.add(new_img_v, filtered_v)
    new_img_s = np.add(new_img_s, filtered_s)


    return new_img_v, new_img_s

def remove_dark(img_v, img_s):

    height = img_v.shape[0]
    minh = int(height * 0.5)
    maxh = height - 20 #remove desktop reflection

    # Divide it into 2 horizontal slides

    n_slides = 3
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

    minx = int(width * 0.3)
    maxx = int(width * 0.4)

    minh = int(height * 0.8)
    maxh = height - 20 #remove desktop reflection


    # First try to get the correct hue values for the tartan
    h_histo = np.histogram(h_channel[minh:maxh, minx:maxx].flatten(), bins=bins)
    h_bin_max = np.argmax(h_histo[0])
    h_min = h_histo[1][h_bin_max]*0.8
    h_max = h_histo[1][h_bin_max + 1]*1.2


    # Threshold saturation and intensity
    h_channel_limited = np.copy(h_channel)
    h_channel_limited[(h_channel < h_min) | (h_channel > h_max)] = 0

    orient = 'x'

    if orient == 'x':
        sobel = cv2.Sobel(h_channel, cv2.CV_64F, 1, 0, ksize=3) # Take the derivative in x
    else:
        sobel = cv2.Sobel(h_channel_limited, cv2.CV_64F, 0, 1, ksize=3)  # Take the derivative in x

    abs_sobel = np.absolute(sobel) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))


    # Threshold saturation and intensity
    s_binary = np.zeros_like(s_channel)
    s_binary[(scaled_sobel >= h_min) & (scaled_sobel <= h_max)] = 1

    kernel = np.ones((dilate, dilate), np.uint8)
    output = cv2.dilate(s_binary,kernel,iterations = 1)


    return output



def sobel_value_x_filter(h_channel, s_channel, v_channel, sobel_thresh=(12, 255), v_thresh=(70, 255), sw_thresh=(0,255)): # was v>50

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


def complex_sobel(h_channel, s_channel, v_channel):
    sobel_v = sobel_value_x_filter(h_channel, s_channel, v_channel, sobel_thresh=(12, 255), v_thresh=(50, 255), sw_thresh=(0, 255)) # 15 was 47
    sobel_s = sobel_saturation_filter(h_channel, s_channel, v_channel, sobel_thresh=(15, 50), v_thresh=(70, 255), sy_thresh=(80, 255))
#    sobel_v = sobel_value_x_filter(h_channel, s_channel, v_channel, sobel_thresh=(35, 255), v_thresh=(25, 255), sw_thresh=(0, 255)) # 15 was 47
#    sobel_s = sobel_saturation_filter(h_channel, s_channel, v_channel, sobel_thresh=(15, 50), v_thresh=(70, 255), sy_thresh=(80, 255))
    mask = np.zeros_like(sobel_s)
    mask[(sobel_v == 1) | (sobel_s == 1)] = 1

    return mask.astype(np.uint8)



## Course filters

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

def gradient_filter(h_channel, s_channel, v_channel, x_thresh=(12,255), y_thresh=(25,255), s_thresh=(100,255), v_thresh=(50,255)):

    grad_x = abs_sobel_thresh(h_channel, s_channel, v_channel, orient='x', thresh=x_thresh)
    grad_y = abs_sobel_thresh(h_channel, s_channel, v_channel, orient='y', thresh=y_thresh)
    c_binary = color_threshold(h_channel, s_channel, v_channel, s_thresh=s_thresh, v_thresh=v_thresh )

    mask = np.zeros_like(h_channel, dtype=np.uint8)

    mask[(((grad_x == 1) & (grad_y == 1)) | (c_binary == 1))] = 1

    return mask

def feed_start(img, l_center, r_center, width=20, height=60):

    ymax = img.shape[0]

    l_int = int(l_center)
    r_int = int(r_center)

    img[ymax-height-20:ymax, l_int-width:l_int+width] = 255
    img[ymax-height-20:ymax, r_int-width:r_int+width] = 255

def super_filter(h_channel, s_channel, v_channel, world):

    mask_gr = gradient_filter(h_channel, s_channel, v_channel, x_thresh=(35, 255), y_thresh=(25, 255),
                              s_thresh=(180, 255), v_thresh=(200, 255))


    mask_gr_1 = gradient_filter(h_channel, s_channel, v_channel, x_thresh=(10, 255), y_thresh=(10, 255),
                              s_thresh=(180, 255), v_thresh=(200, 255))

    # 1mask_gr = gradient_filter(h_channel, s_channel, v_channel, x_thresh=(50, 255), y_thresh=(25, 255), s_thresh=(180, 255), v_thresh=(200, 255))
    #mask_gr = gradient_filter(h_channel, s_channel, v_channel, x_thresh=(15, 255), y_thresh=(25, 255), s_thresh=(120, 255), v_thresh=(150, 255))
    mask_so = complex_sobel(h_channel, s_channel, v_channel)

    tartan = color_filter(h_channel, s_channel, v_channel, orient='x', h_thresh=(100, 130), hx_thresh=(50,130), dilate=30)

    mask_gr_t = np.zeros_like(tartan)
    mask_gr_t[((tartan == 1) & (mask_gr == 1))] = 1


    mask_so_t = np.zeros_like(tartan)
    mask_so_t[((tartan == 1) & (mask_so == 1))] = 1




    #if world.frame == -1:
    world.add_debug("Dates", mask_gr*255)

    return [['gr', mask_gr*255], ['gr1', mask_gr_1*255],['so', mask_so*255], ['grt', mask_gr_t*255], ['sot', mask_so_t*255]]
    #return [['gr', mask_gr*255],  ['gr1', mask_gr_1*255]]
    #return [['gr', mask_so_t * 255]]
    #return [['grt', mask_gr_t*255], ['sot', mask_so_t*255]]


### process_image processes a original image by
#
#   Undostorting
#   Filtering
#   Warping
#
# returns undistorted, [array of image filtered with different filters]

def process_image(image, world):

    # Get some data

    cal = world.calibration

    img_size = (image.shape[1], image.shape[0])
    undistorted = cv2.undistort(image, cal.mtx, cal.dist, None, cal.mtx)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    eqs = cv2.equalizeHist(s_channel)
    eqv = cv2.equalizeHist(v_channel)

    eqs = cv2.GaussianBlur(eqs, (7, 7), 0., 0.)
    eqv = cv2.GaussianBlur(eqv, (7, 7), 0., 0.)

    #world.add_debug("Intensity", eqv)
    #world.add_debug("Saturation", eqs)
    eqv, eqs = remove_dark(eqv, eqs)


    filtered = super_filter(h_channel, eqs, eqv, world)
    # Generate warped images

    # bw_lines = cv2.warpPerspective(filtered, cal.M, img_size, flags=cv2.INTER_LINEAR) # Warp it to get top view

    bw_lines = list(map(lambda f: [f[0], cv2.warpPerspective(f[1], cal.M, img_size, flags=cv2.INTER_LINEAR)], filtered))
    #world.add_debug("Intensity", bw_lines[0][1])
    #world.add_debug("Saturation", bw_lines[1][1])

    return undistorted, bw_lines, eqv

### MEASUREMENT FUNCTIONS
#
#   These functions read data from the world, analyze it and buld them
#       as a measurement.
#
#   Some may use actual knowledge of the world, for example the known_fit algorithm.
#
#   Others just ignore the world ans are pure blind. Ex. the sliding_windows algorithm


### Auxiliary function to generate a windows mask
def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
    max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
    return output


### window_centroids looks for the center of the windows
#
#   warped is the filtered, warped image
#   window_width and height
#   margin is th margin to search
#   min_points is the minimum value of the convolution to be included
#       to eliminat spurious results
#
# returns
#   left and right centroid list

def find_window_centroids(warped, window_width, window_height, margin, min_points, limits=(300, 1050),
                          center_limit=150):
    offset = window_width / 2
    window_centroids_left = []  # Store the left, window centroid positions per level
    window_centroids_right = []  # Store the right window centroid positions per level
    window = np.ones(window_width)  # Create our window template that we will use for convolutions

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template

    # Sum quarter bottom of image to get slice, could use a different ratio
    # l_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, limits[0]:int(warped.shape[1] / 2)], axis=0)
    l_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, limits[0]:int(warped.shape[1] / 2) - center_limit], axis=0)
    l_conv = np.convolve(window, l_sum)
    l_center = np.argmax(l_conv) - window_width / 2 + limits[0]
    # r_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, int(warped.shape[1] / 2):limits[1]], axis=0)
    r_sum = np.sum(warped[int(3 * warped.shape[0] / 4):, int(warped.shape[1] / 2) + center_limit:limits[1]], axis=0)
    r_conv = np.convolve(window, r_sum)
    r_center = np.argmax(r_conv) - window_width / 2 + int(warped.shape[1] / 2) + center_limit

    l_value = l_conv[int(l_center + offset - limits[0])]
    r_value = r_conv[int(r_center + offset) - int(warped.shape[1] / 2) - center_limit]

    # Add what we found for the first layer

    old_lcenter = l_center
    old_rcenter = r_center
    window_centroids_left.append(l_center)
    window_centroids_right.append(r_center)

    old_old_lcenter = l_center
    old_lcenter = l_center
    old_old_rcenter = r_center
    old_rcenter = r_center

    l_factor = 1
    r_factor = 1

    l_side = 0
    r_side = 0

    # Go through each layer looking for max pixel locations

    for level in range(1, (int)(warped.shape[0] / window_height)):  # Era 1
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(
            warped[int(warped.shape[0] - (level + 1) * window_height):int(warped.shape[0] - level * window_height),
            :],
            axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window

        l_min_index = int(max(old_lcenter + offset - margin, 0))
        l_max_index = int(max(min(old_lcenter + offset + margin, warped.shape[1]), window_width))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
        l_value = conv_signal[int(l_center + offset)]

        # Find the best right centroid by using past right center as a reference
        r_min_index = int(min(max(old_rcenter + offset - margin, 0), warped.shape[1]-window_width))
        r_max_index = int(min(old_rcenter + offset + margin, warped.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
        r_value = conv_signal[int(r_center + offset)]

        if abs(l_value) >= min_points and l_center < (warped.shape[1] / 2):   #and abs(l_center - old_lcenter) < window_width*1.5:
            window_centroids_left.append(l_center)
            old_old_lcenter = old_lcenter
            old_lcenter = l_center
            l_factor = 1

        else:
            l_center = old_lcenter + (old_lcenter - old_old_lcenter)
            window_centroids_left.append(l_center)
            #l_center = old_lcenter + (old_lcenter - old_old_lcenter)
            old_old_lcenter = old_lcenter
            old_lcenter = l_center
            l_factor = 1

        if abs(r_value) >= min_points and r_center > warped.shape[1] / 2 :  #and abs(r_center - old_rcenter) < window_width*1.5:
            window_centroids_right.append(r_center)
            old_old_rcenter = old_rcenter
            old_rcenter = r_center
            r_factor = 1

        else:
            r_center = old_rcenter + (old_rcenter - old_old_rcenter)
            window_centroids_right.append(r_center)
            #r_center = old_rcenter + (old_rcenter - old_old_rcenter)
            old_old_rcenter = old_rcenter
            old_rcenter = r_center

            r_factor = 1

    return window_centroids_left, window_centroids_right

### sliding_windows
#
#   Uses the sliding windows method to select line points
#   returns updated lines
#
#   warped_image is the filtered watped image
#   window_width and window_height are the window width and height
#   margin is the margin to search left and right in pizels
#
# returns
#   type, left_fit, right_fit, left_fit_cr, right_fit_cr, l_points, r_points
#
#   type is always 0 here


def sliding_window_fit(filtered, world):  # Let's go

    filter_name = filtered[0]
    warped_image = filtered[1]

    maxy = warped_image.shape[0] - 1

    window_width = world.window_width
    window_height = world.window_height
    margin = world.window_margin
    min_points = world.min_points

    if world.working_belief is not None and world.skipped < world.num_windows and False:
        lcenter = world.working_belief.left_data.get_base_x()
        rcenter = world.working_belief.right_data.get_base_x()
        feed_start(warped_image, lcenter, rcenter, width=20, height=30)


    window_centroids_left, window_centroids_right = find_window_centroids(warped_image, window_width, window_height,
                                                                          margin, min_points)

    # now we must compute the points of left and right lines

    nonzero = warped_image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = []
    right_lane_inds = []

    l_points = np.zeros_like(warped_image)
    r_points = np.zeros_like(warped_image)

    # Left lane windows

    for level in range(0, len(window_centroids_left)):
        win_y_low = warped_image.shape[0] - (level + 1) * window_height
        win_y_high = warped_image.shape[0] - level * window_height

        if window_centroids_left[level] >= 0:
            l_mask = window_mask(window_width, window_height, warped_image, window_centroids_left[level], level)
            l_points[(l_points == 255) | ((l_mask == 1))] = 255
            win_xleft_low = window_centroids_left[level] - window_width / 2
            win_xleft_high = window_centroids_left[level] + window_width / 2
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                nonzerox < win_xleft_high)).nonzero()[0]
            left_lane_inds.extend(good_left_inds)

    # right lane windows

    for level in range(0, len(window_centroids_right)):
        win_y_low = warped_image.shape[0] - (level + 1) * window_height
        win_y_high = warped_image.shape[0] - level * window_height

        if window_centroids_right[level] > 0:
            # Add graphic points from window mask here to total pixels found
            r_mask = window_mask(window_width, window_height, warped_image, window_centroids_right[level], level)
            r_points[(r_points == 255) | ((r_mask == 1))] = 255
            win_xright_low = window_centroids_right[level] - window_width / 2
            win_xright_high = window_centroids_right[level] + window_width / 2
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                nonzerox < win_xright_high)).nonzero()[0]
            right_lane_inds.extend(good_right_inds)

    # Compute pixels and generate ajustment polygons, leftx and lefty are coordinates of points used
    # They are number arrays

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if len(leftx) > 0:
        left_lane = Lane_Measure.new_lane_measure_from_data(leftx, maxy-lefty, l_points, side='left',
                                                            filter_x=filter_name,
                                                            method='windows', xm=world.calibration.xm,
                                                            ym=world.calibration.ym)
    else:
        left_lane = None

    if len(rightx) > 0:
        right_lane = Lane_Measure.new_lane_measure_from_data(rightx, maxy-righty, r_points, side='right',
                                                             filter_x=filter_name,
                                                             method='windows', xm=world.calibration.xm,
                                                             ym=world.calibration.ym)
    else:
        right_lane = None

    if left_lane is None or right_lane is None:
        return None

    measure = Measure(left_lane, right_lane, warped_image)

    return measure

### Once known the fits we may look around a margin from the old fits
#       and update it.
#       No need to look for sliding windows or similar thongs
#
#   warped_image the filtered and warped imade
#   margin pixels around the current fit to look for points
#   l_line, r_line line structures representing current fits
#
# returns
#   type, left_fit, right_fit, left_fit_cr, right_fit_cr, l_points, r_points
#
#   type is always 1 here

def known_lines_fit(filtered, world):

    filter_name = filtered[0]
    warped_image = filtered[1]
    margin = world.margin

    maxy = warped_image.shape[0]

    current_belief = world.working_belief

    if current_belief is None:
        return None, None

    else:
        left_fit = current_belief.left_data.fit
        right_fit = current_belief.right_data.fit

    if world.skipped < world.num_windows and False:
        feed_start(warped_image, left_fit.value(0.0), right_fit.value(0.0), width=20, height=20)

    nonzero = warped_image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    inv_nonzeroy = maxy - nonzeroy

    model_left_fit = left_fit
    model_right_fit = right_fit

    left_lane_inds = []
    right_lane_inds = []

    left_lane_inds = (
    (nonzerox > (left_fit.value(inv_nonzeroy) - margin)) & (nonzerox < (left_fit.value(inv_nonzeroy) + margin)))
    right_lane_inds = (
    (nonzerox > (right_fit.value(inv_nonzeroy) - margin)) & (nonzerox < (right_fit.value(inv_nonzeroy) + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Points used to draw all the left and right windows. They are computed from old fit
    #   So no problem if not found the fit
    #

    l_points = np.zeros_like(warped_image)
    r_points = np.zeros_like(warped_image)

    # Generate x and y values for plotting
    ploty = maxy - np.linspace(0, warped_image.shape[0] - 1, warped_image.shape[0])

    left_fitx = left_fit.value(ploty)
    right_fitx = right_fit.value(ploty)

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, maxy - ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, maxy - ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))

    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, maxy - ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, maxy - ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    cv2.fillPoly(l_points, np.int_([left_line_pts]), 255)
    cv2.fillPoly(r_points, np.int_([right_line_pts]), 255)

    if len(leftx) > 0:
        left_lane = Lane_Measure.new_lane_measure_from_data(leftx, maxy-lefty, l_points, filter_x=filter_name,
                                                            method='fit', old_fit = left_fit,
                                                            xm=world.calibration.xm, ym=world.calibration.ym)
    else:
        left_lane = None

    if len(rightx) > 0:
        right_lane = Lane_Measure.new_lane_measure_from_data(rightx, maxy-righty, r_points, filter_x=filter_name,
                                                             method='fit', old_fit=right_fit,
                                                             xm=world.calibration.xm, ym=world.calibration.ym)
    else:
        right_lane = None

    measure = Measure(left_lane, right_lane, warped_image)

    return measure

### DISPLAY AND PROCESS FUNCTIONS

### plot_lines
#
#   generates a color image a 1/3 the size of the original with the left
#   and right points and the look for polygons and the fit
#
#   Returns the color image
def plot_lines(warped_image, belief, scale, world):

    # Draw the results

    l_points = belief.measures[0].left_data.selection_points
    r_points = belief.measures[0].right_data.selection_points
    maxn = warped_image.shape[0] - 1
    ploty = maxn - np.linspace(0, warped_image.shape[0] - 1, warped_image.shape[0])

    if r_points is not None and l_points is not None:
        template = np.array(r_points + l_points, np.uint8)  # add both left and right window pixels together
        zero_channel = np.zeros_like(template)  # create a zero color channel
        template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)  # make window pixels green

        warpage = np.array(cv2.merge((warped_image, warped_image, warped_image)), np.uint8)  # making the original road pixels 3 color channels


        if len(belief.measures) > 0:

            measure = belief.measures[0]

            if measure.left_data.fit is not None:
                left_fitx = measure.left_data.fit.value(ploty)
                left_points = np.array([left_fitx, maxn - ploty], dtype=np.int32)
                left_points = left_points.transpose()
                thickness = int(np.sqrt(measure.left_data.sigma2)*2)
                cv2.polylines(template, [left_points], 0, (255, 255, 0), thickness=thickness, lineType=cv2.LINE_AA)

            if measure.right_data.fit is not None:
                right_fitx = measure.right_data.fit.value(ploty)
                right_points = np.array([right_fitx, maxn - ploty], dtype=np.int32)
                right_points = right_points.transpose()
                thickness = int(np.sqrt(measure.right_data.sigma2)*2)
                cv2.polylines(template, [right_points], 0, (255, 255, 0), thickness=thickness, lineType=cv2.LINE_AA)

            # Draw found lines over image

        left_fitx = belief.left_data.get_x(ploty)
        right_fitx = belief.right_data.get_x(ploty)

        left_points = np.array([left_fitx, maxn - ploty], dtype=np.int32)
        right_points = np.array([right_fitx, maxn - ploty], dtype=np.int32)
        left_points = left_points.transpose()
        right_points = right_points.transpose()

        left_thickness = int(math.sqrt(belief.left_data.sigma2) * 2)
        right_thickness = int(math.sqrt(belief.right_data.sigma2) * 2)

        cv2.polylines(template, [left_points], 0, (0, 0, 255), thickness=left_thickness, lineType=cv2.LINE_AA)
        cv2.polylines(template, [right_points], 0, (0, 0, 255), thickness=right_thickness, lineType=cv2.LINE_AA)

        # output = warpage
        output = cv2.addWeighted(warpage, 1.0, template, 0.5, 0.0)  # overlay the orignal road image with window results




        # If there are old_fits draw them in orange

#        if belief.left_data.old_fit is not None:
#            left_fitx = belief.left_data.old_fit.value(ploty)
#            left_points = np.array([left_fitx, maxn - ploty], dtype=np.int32)
#            left_points = left_points.transpose()
#            cv2.polylines(output, [left_points], 0, (128, 128, 255), thickness=4, lineType=cv2.LINE_AA)


#       if belief.right_data.old_fit is not None:
#            right_fitx = belief.right_data.old_fit.value(ploty)
#            right_points = np.array([right_fitx, maxn - ploty], dtype=np.int32)
#            right_points = right_points.transpose()
#            cv2.polylines(output, [right_points], 0, (128, 128, 255), thickness=4, lineType=cv2.LINE_AA)





        if belief.center_lane is not None and belief.center_lane.fit is not None:
            right_fitx = belief.center_lane.fit.value(ploty)
            right_points = np.array([right_fitx, maxn - ploty], dtype=np.int32)
            right_points = right_points.transpose()
            cv2.polylines(output, [right_points], 0, (0, 0, 255), thickness=4, lineType=cv2.LINE_AA)




        if scale != 1 :
            w = int(output.shape[1] / scale)
            h = int(output.shape[0] / scale)

            small_output = np.zeros((h, w, 3), dtype=np.uint8)

            cv2.resize(output, (w, h), small_output)
        else:
            small_output = np.copy(output)

    else:
        w = int(warped_image.shape[1] / scale)
        h = int(warped_image.shape[0] / scale)

        small_output = np.zeros((h, w, 3), dtype=np.uint8)

    return small_output

def dummy_lines(warped_image):

    w = int(warped_image.shape[1] / 4)
    h = int(warped_image.shape[0] / 4)

    small_output = np.zeros((h, w, 3), dtype=np.uint8)

    return small_output

### Generates a polygon needs image height

def get_polygon(warped_image, belief):
    # Draw found lines over image

    maxn = warped_image.shape[0]-1

    ploty = maxn - np.linspace(0, warped_image.shape[0] - 20, warped_image.shape[0])
    left_fitx = belief.left_data.get_x(ploty)
    right_fitx = belief.right_data.get_x(ploty)

    pts_left = np.array([np.transpose(np.vstack([left_fitx, maxn - ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, maxn - ploty])))])
    pts = np.hstack((pts_left, pts_right))

    return pts


### Draws info in image
#   Draws lane area
#   Draws data
#
def build_edited_image(image, belief, insert, font, world):

    frame = belief.frame

    ymax = image.shape[0]

    l_line = belief.left_data
    r_line = belief.right_data

    warped = cv2.warpPerspective(image, world.calibration.M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)


    #poly = get_polygon(image, l_line.current_fit, r_line.current_fit)
    poly = get_polygon(image, belief)


    # Create an image to draw the lines on
    color_warp = np.zeros_like(warped).astype(np.uint8)
    # color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([poly]), (0, 255, 0))


    # Draw Center line

    maxn = color_warp.shape[0] - 1
    ploty = maxn - np.linspace(0, color_warp.shape[0] - 20, color_warp.shape[0])

    if belief.center_lane is not None and belief.center_lane.fit is not None:
        right_fitx = belief.center_lane.fit.value(ploty)
        right_points = np.array([right_fitx, maxn - ploty], dtype=np.int32)
        right_points = right_points.transpose()
        cv2.polylines(color_warp, [right_points], 0, (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))



    out_img = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

    # Now write the data on the image

    rad = world.get_avg_radius()

    divergence = belief.divergence(0, ymax)
    lane_offset = belief.position_w()
    lane_width = world.calibration.xm * world.lane_width
    lane_width_error = np.sqrt(world.lane_sigma2) * world.calibration.xm
    left_radius = belief.left_data.radius
    right_radius = belief.right_data.radius

    if frame != -1:
        line_1 = "Frame {} Radius {:0.0f} m  Divergence {:0.2f}%".format(frame, rad, divergence * 100)
    else:
        line_1 = "Filter {} Radius {:0.0f} m  Divergence {:0.2f}%".format(belief.left_data.filter, rad, divergence * 100)

    if lane_offset < 0:
        dir_offset = "right"
    else:
        dir_offset = "left"

    line_2 = "Width {:0.2f} m Position {:0.2f} m to the {}".format(lane_width,  abs(lane_offset), dir_offset)
    line_3 = "{:0.0f} - {:0.0f} - {:0d}".format(left_radius, right_radius, world.skipped)
    line_4 = "Filter {} - {} ".format(belief.left_data.filter, belief.right_data.filter)

    cv2.putText(out_img, line_1, (10, 20), font, 1.5, (255, 255, 255), 2)
    cv2.putText(out_img, line_2, (10, 50), font, 1.5, (255, 255, 255), 2)
    #cv2.putText(out_img, line_3, (940, 220), font, 1.5, (255, 255, 255), 2)
    cv2.putText(out_img, line_4, (940, 220), font, 1.5, (255, 255, 255), 2)

    if insert is not None:
        x_offset = 940
        y_offset = 20
        out_img[y_offset:y_offset + insert.shape[0], x_offset:x_offset + insert.shape[1]] = insert

    return out_img

## resize resizes an imags
def resize(img, scale):
    if scale != 1:
        w = int(img.shape[1] / scale)
        h = int(img.shape[0] / scale)

        output = np.zeros((h, w, 3), dtype=np.uint8)

        cv2.resize(img, (w, h), output)
    else:
        output = img

    return output

def insert_at(big_img, small_img, x, y):

    x_offset = x
    y_offset = y
    big_img[y_offset:y_offset + small_img.shape[0], x_offset:x_offset + small_img.shape[1]] = small_img


## Process all images
#
#
def process_folder(world, folder, log_folder=None):

    for name in os.listdir(folder):
        process_an_image(world, folder, name, log_folder=log_folder)


## Process an image
#
# processes an image with the Sliding Windows system
#
#


def process_an_image(world, folder, name, log_folder=None):

    image = cv2.imread(folder + name)

    process_una_imatge(world, image, name, log_folder=log_folder)

def process_una_imatge(world, image, name, log_folder=None):

    world.reset()

    font = cv2.FONT_HERSHEY_PLAIN
    win = cv2.namedWindow('image')

    world.frame = -1

    if image is not None:
        undistorted, filtereds, debug_img = process_image(image, world)
        measures = list(map(lambda f: sliding_window_fit(f, world), filtereds))
        if measures is not None:
            world.update(measures)

        belief = world.working_belief
        warped = world.working_belief.warped_image
        color_warped = plot_lines(warped, belief, 4, world)
        edited_frame = build_edited_image(undistorted, belief, color_warped, font, world)

        y = 260

        for entry in world.debug_list:
            img = entry[2]
            if len(img.shape) == 3:
                color_img = img
            else:
                color_img = np.stack((img, img, img), axis=2)
            img_small = resize(color_img, 4)
            insert_at(edited_frame, img_small, 940, y)
            y += 200



        if log_folder is not None:
            f = "{}/{}".format(log_folder, name)
            cv2.imwrite(f, edited_frame)

        #cv2.imshow("Warped", color_warped)
        cv2.imshow(name, edited_frame)  # was out_img - may be warped to analysis
        while True:
            k = cv2.waitKey(20) & 0xff
            if k == 27:
                break
            else:
                continue

    cv2.destroyAllWindows()


## Process video
#
# filename is the name of the folder
# first is first frame to process
# last is last frame to process
# log_folder if not None is a folder where to store each original frame
# output_video if not None is the pathname where the edited viceo will be written


def process_video(world, filename, first=1, last=100000, log_folder=None, output_video=None):

    world.reset()

    font = cv2.FONT_HERSHEY_PLAIN
    win = cv2.namedWindow('Video')

    old_filtereds = []

    cap = cv2.VideoCapture(filename)  # Careful images are BGR

    frame = 0
    ret = True

    if output_video is not None:
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        writer = cv2.VideoWriter(output_video, fourcc, 24,
                                 (1280, 720), True)

    while ret and frame < first:  # Skip over the non wanted frames
        ret, image = cap.read()
        frame = frame + 1

    while ret and frame <= last:

        # Process the frame
        world.debug_list = []

        world.frame = frame

        undistorted, new_filtereds, debug_img = process_image(image, world)

        ## Compute new filtereds as sum of two images

        if len(old_filtereds) == 0:
            old_filtereds = new_filtereds

        filtereds = []

        # Add Points
        for old, new in zip(old_filtereds, new_filtereds):
            new_warped = cv2.add(old[1], new[1])
            filtereds.append([new[0], new_warped])  # Average was new_warped

        old_filtereds = new_filtereds

        # Advance the world to actual frame


        if world.working_belief is not None:
            if world.skipped < world.use_windows:
                world.advance(to_frame=frame)
                measures = list(map(lambda f: known_lines_fit(f, world), filtereds))
            else:
                world.advance(to_frame=frame)
                measures = list(map(lambda f: sliding_window_fit(f, world), filtereds))
        else:
            #world.advance(to_frame=frame)
            measures = list(map(lambda f: sliding_window_fit(f, world), filtereds))

        # UPDATE THE WORLD

        if measures is not None:
            world.update(measures)
        else:
            world.skipped += 1


        belief = world.working_belief

        # This is just in case frst frame is very bad and has been rejected

        if belief is None:
            ret, image = cap.read()
            frame = frame + 1
            world.frame = frame
            continue

        # Display data

        warped = world.working_belief.warped_image
        color_warped = plot_lines(warped, belief, 4, world)
        edited_frame = build_edited_image(undistorted, belief, color_warped, font, world)

        y = 260

        # debug list is for passing information (images) from deep in the filters

        for entry in world.debug_list:
            img = entry[2]
            color_img = np.stack((img, img, img), axis=2)
            img_small = resize(color_img, 4)
            insert_at(edited_frame, img_small, 940, y)
            y += 200

        # Save images in folder if needed

        if log_folder is not None:
            f = "{}/{}.jpg".format(log_folder, frame)
            cv2.imwrite(f, image)
            f_mask = "{}/{}_mask.jpg".format(log_folder, frame)
            cv2.imwrite(f_mask, color_warped)

        # Write frame to video output
        if output_video is not None:
            writer.write(edited_frame)

        cv2.imshow('Video', edited_frame)  # was out_img - may be warped to analysis

        ret, image = cap.read()
        frame = frame + 1
        world.frame = frame

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        else:
            continue

    if output_video is not None:
        writer.release()

    cv2.destroyAllWindows()

### AUXILIARY FUNCTIONS




### MAIN PROGRAM

### Loads wide_dist_pickle.p. Reads and processes image

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30.0 / 720  # meters per pixel in y dimension
xm_per_pix = 3.7 / 900  # meters per pixel in x dimension

# Video files
circuit_videos = ['project_video.mp4', 'challenge_video.mp4', 'harder_challenge_video.mp4']
video_folder = './images/videos/'

cal_data = pickle.load(open("./wide_dist_pickle.p", "rb"))
mtx = cal_data["mtx"]
dist = cal_data["dist"]
circuit = 0 # Which circuit to do (0, 1 or 2)

if circuit == 0:  # project_video
    M, Minv, xm_per_pix = perspective(f=1.280, h=455, shrink=0.15, xmpp=xm_per_pix)  # For video 1
    margin = 80

elif circuit == 1:  # challenge_video
    M, Minv, xm_per_pix = perspective(f=1.393, h=507, shrink=0.15, xmpp=xm_per_pix)  # For video 2
#    M, Minv, xm_per_pix = perspective(f=1.393, h=475, shrink=0.15, xmpp=xm_per_pix)  # For video 2
    margin = 80  # for video 2

elif circuit == 2:  # harder_challenge_video
    M, Minv, xm_per_pix = perspective(f=1.263, h=550, shrink=0.15, xmpp=xm_per_pix)  # For video 3
    margin = 50

# Create out world.

cal = Calibration(mtx, dist, M, Minv, xm_per_pix, ym_per_pix)
world = World(cal)
world.margin = margin
world.max_skipped = 20
world.num_windows = 5
world.speed = 24
world.speed_sigma2 = 10

video_pathname = video_folder + circuit_videos[circuit]

process_video(world, video_pathname, first=1)#, output_video="./images/videos/challenge_video_out.avi")
#process_an_image(world, "./images/log_images2/", "128.jpg")
#process_folder(world, "./images/log_images3/") #, log_folder="./images/test_images_out")








