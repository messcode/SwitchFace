"""
A collection of function handling faces.
"""

import cv2
import dlib
import numpy

PREDICTOR_PATH = "predictor_data\\shape_predictor_68_face_landmarks.dat"
SCALE_FACTOR = 1 
FEATHER_AMOUNT = 11

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                               RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)
# ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS + JAW_POINTS)                              

# Points from the second image to overlay on the first. The convex hull of each
# element will be overlaid.
OVERLAY_POINTS = [
    LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
    NOSE_POINTS + MOUTH_POINTS,
]
# OVERLAY_POINTS = [LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS + JAW_POINTS]

# Amount of blur to use during colour correction, as a fraction of the
# pupillary distance.
COLOUR_CORRECT_BLUR_FRAC = 0.6                               

# Returns the default face detector
detector = dlib.get_frontal_face_detector()  

# This object is a tool that takes in an image region containing some object 
# and outputs a set of point locations that define the pose of the object. 
# The classic example of this is human face pose prediction, where you take 
# an image of a human face as input and are expected to identify the locations 
# of important facial landmarks such as the corners of the mouth and eyes, 
# tip of the nose, and so forth.
predictor = dlib.shape_predictor(PREDICTOR_PATH)

class ImproperNumber(Exception):
    pass

def get_landmarks(img):
    """
    Get face landmarks from a rectangle region in image.
    Args:
        img: a cv2 image object.
        rect:  
    Retrun:
         1 * 68 numpy matrix corresponding to landmark points. 
    """
    rects = detector(img, 1) # 1 is upsampling factor.
    return [numpy.matrix([[p.x, p.y] for p in predictor(img, rect).parts()]) for rect in rects] 
    
def annotate_landmarks(img, landmarks, font_scale = 0.4):
    """
    Annotate face landmarks on image. 
    Args:
        img: a cv2 image object.
        landmarks: numpy matrix consisted of points.
    Return:
        annotated image.
    """
    img = img.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(img, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=font_scale,
                    color=(0, 0, 255))
        cv2.circle(img, pos, 3, color=(0, 255, 255))
    return img        

    
def draw_convex_hull(img, points, color):  
    """
    Draw convex hull on img. Figure img will be changed after calling this function.
    """
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(img, points, color=color)

def partial_blur(img, points, kenel_size = 9, type = 1):
    """
    Partial Gaussian blur within convex hull of points.
    Args:
        type = 0 for Gaussian blur
        type = 1 for average blur
    """
    points = cv2.convexHull(points)
    copy_img = img.copy()
    black = (0, 0, 0)
    if type:  
        cv2.blur(img, (kenel_size, kenel_size)) 
    else:
        cv2.GaussianBlur(img, (kenel_size, kenel_size), 0)
    cv2.fillConvexPoly(copy_img, points, color = black)
    for row in range(img.shape[:2][0]):
        for col in range(img.shape[:2][1]):
            if numpy.array_equal(copy_img[row][col], black):
                copy_img[row][col] = blur_img[row][col] 
    return copy_img
    



def get_face_mask(img, landmarks):
    """
    Get face mask matrix, mask area = 1.
    """
    img = numpy.zeros(img.shape[:2], dtype=numpy.float64)

    for group in OVERLAY_POINTS:
        draw_convex_hull(img,
                         landmarks[group],
                         color=1)

    img = numpy.array([img, img, img]).transpose((1, 2, 0))

    img = (cv2.GaussianBlur(img, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    img = cv2.GaussianBlur(img, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return img
    
def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:
        sum ||s*R*p1,i + T - p2,i||^2
    is minimized.
    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)

    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    # normalization
    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2  

    U, S, Vt = numpy.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         numpy.matrix([0., 0., 1.])])      
                         
def warp_im(im, M, dshape):
    """
    Affine transformation with matrix M to dshape.
    """
    output_im = numpy.zeros(dshape, dtype=im.dtype)  # zero matrix
    cv2.warpAffine(im,
                   M[:2], # shape of M
                   (dshape[1], dshape[0]),
                   dst = output_im,
                   borderMode = cv2.BORDER_TRANSPARENT,
                   flags = cv2.WARP_INVERSE_MAP)
    return output_im                                               
    
def correct_colours(im1, im2, landmarks1):
    """
    Attempt to change the colouring of im2 to match that of im1. 
    It does this by dividing im2 by a gaussian blur of im2,  and then multiplying 
    by a gaussian blur of im1.
    """
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * numpy.linalg.norm(
                              numpy.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                              numpy.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(numpy.float64) * im1_blur.astype(numpy.float64) /
                                                im2_blur.astype(numpy.float64))  
                                                
def align_face(src, landmark_src, dest, landmark_dest):
    """
    Align face  in src to dest image.
    """ 
    M = transformation_from_points(landmark_dest[ALIGN_POINTS], landmark_src[ALIGN_POINTS])
    mask = get_face_mask(src, landmark_src)
    warped_mask = warp_im(mask, M, src.shape)
    combined_mask = numpy.max([get_face_mask(dest, landmark_dest), warped_mask], axis = 0)
    warped_src = warp_im(src, M, dest.shape)
    warped_corrected_src = correct_colours(dest, warped_src, landmark_dest)
    
    output_im = dest * (1 - combined_mask) + warped_corrected_src * combined_mask
    return output_im.astype(numpy.uint8) 
     
def to_uint8(img):
    """
    Cast data type of numpy array to unsigned int8.  
    """
    return img.astype(numpy.uint8)
    
def switch_face(img_path):
    """
    Switch faces in image.
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    landmarks = get_landmarks(img)
    if len(landmarks) < 1:
        raise ImproperNumber("Faces detected is less than 2!")
    if len(landmarks) > 2:
        raise ImproperNumber("Faces detected is more than 2!")
    
    output = align_face(img, landmarks[0], img, landmarks[1])
    output = align_face(img, landmarks[1], output, landmarks[0])
    return output
