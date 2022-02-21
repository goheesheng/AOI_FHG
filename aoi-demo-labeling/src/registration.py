import numpy as np
import cv2
import imreg_dft as ird


def register_images_ft(tmpl, img, return_matrix=False):
    """ Image registration using phase shift of Fourier Transform """

    t_h, t_w = tmpl.shape[:2]
    if tmpl.shape != img.shape:
        img = cv2.resize(img, (t_w, t_h), interpolation=cv2.INTER_AREA)

    center = (t_w // 2, t_h // 2)

    reg_result = ird.similarity(tmpl, img, numiter=1, order=3,
                                constraints={
                                    'angle': [0, 3],
                                    'scale': [1.0, 0.03],
                                    'tx': [0, 10],
                                    'ty': [0, 10]
                                })

    if reg_result['success'] > 0.2:
        t_vec = reg_result['tvec']
        t_angle = reg_result['angle']
        t_scale = reg_result['scale']

        mat = cv2.getRotationMatrix2D(center, -t_angle, t_scale)
        mat[:, 2] = mat[:, 2] + t_vec[::-1]        
    else:
        mat = None                

    return mat


def register_images_ecc(tmpl, img, return_matrix=False):
    """ Image registration using enhanced correlation coefficient (ECC) maximization

    Citation:
    Georgios D Evangelidis and Emmanouil Z Psarakis. Parametric image alignment using enhanced correlation coefficient maximization. 
    Pattern Analysis and Machine Intelligence, IEEE Transactions on, 30(10):1858–1865, 2008.
    """

    t_h, t_w = tmpl.shape[:2]
    if tmpl.shape != img.shape:
        img = cv2.resize(img, (t_w, t_h), interpolation=cv2.INTER_AREA)

    # choose between: cv2.MOTION_TRANSLATION, cv2.MOTION_EUCLIDEAN, cv2.MOTION_AFFINE
    warp_mode = cv2.MOTION_EUCLIDEAN
    mat = np.eye(2, 3, dtype=np.float32)
    n_iter = 10000  # number of ECC optimization iterations.
    eps = 1e-7  # terminate if optimization increment lower than this threshold
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                n_iter, eps)

    cc, mat = cv2.findTransformECC(img, tmpl,
                                   mat,
                                   warp_mode,
                                   criteria,
                                   None, 5)

    return mat


def register_images_fm(tmpl, img, return_matrix=False):
    """ Image registration using feature matching """

    t_h, t_w = tmpl.shape[:2]
    if tmpl.shape != img.shape:
        img = cv2.resize(img, (t_w, t_h), interpolation=cv2.INTER_AREA)

    main_thresh = 0.75
    feat_detector = cv2.AKAZE_create()
    matcher = cv2.BFMatcher()

    kps1, des1 = feat_detector.detectAndCompute(img, None)
    kps2, des2 = feat_detector.detectAndCompute(tmpl, None)

    matches = matcher.knnMatch(des1, des2, k=2)

    good_matches = filter_matches(matches, main_thresh)

    if len(good_matches) > 2:
        points1 = np.float32(
            [kps1[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        points2 = np.float32(
            [kps2[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        mat, _ = cv2.estimateAffine2D(points1, points2)
    else:
        mat = None

    return mat


def filter_matches(matches, thresh):
    """ removes ambiguous matches """
    good_matches = []
    for m, n in matches:
        if m.distance < thresh * n.distance:
            good_matches.append([m])

    return good_matches