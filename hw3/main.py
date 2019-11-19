import numpy as np
import cv2
import time

# u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
# this function should return a 3-by-3 homography matrix
def solve_homography(u, v):
    N = u.shape[0]
    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')
    # A = np.zeros((2*N, 8))
    # if you take solution 2:
    A = np.zeros((2*N, 9))
    b = np.zeros((2*N, 1))
    H = np.zeros((3, 3))

    # TODO: some magic
    dst = convert_to_homography_param(v.T)
    src = convert_to_homography_param(u.T)
    ## adjust the points
    src, c1 = normalize(src)
    dst, c2 = normalize(dst)
    ## contruct matrix A
    for i in range(N):
        A[2 * i] = np.array([-src[0][i], -src[1][i], -1, 0, 0, 0, dst[0][i] * src[0][i], dst[0][i] * src[1][i], dst[0][i]])
        A[2 * i + 1] = np.array([0, 0, 0, -src[0][i], -src[1][i], -1, dst[1][i] * src[0][i], dst[1][i] * src[1][i], dst[1][i]])
    ## use svd to get homography matrix
    U, S, V = np.linalg.svd(A)
    H = V[-1].reshape(3, 3)
    H = np.dot(np.linalg.inv(c2), np.dot(H, c1))
    return H / H[2, 2]

def normalize(point_list):
    # type: (np.ndarray) -> (np.ndarray, np.ndarray)
    """
    :param point_list: point list to be normalized
    :return: normalization results
    """
    m = np.mean(point_list[:2], axis=1)
    max_std = max(np.std(point_list[:2], axis=1)) + 1e-9
    c = np.diag([1 / max_std, 1 / max_std, 1])
    c[0][2] = -m[0] / max_std
    c[1][2] = -m[1] / max_std
    return np.dot(c, point_list), c

def convert_to_homography_param(point_list):
    """
    :return: matrix of homography param (3 x N). N = width x height.
    """
    return np.vstack((point_list, np.ones((1, point_list.shape[1]))))

def bilinear_interpolation(h_origin, w_origin, corners):
    """
    :return: pixel value after interpolation
    """
    ## order points by h then w
    corners = sorted(corners)
    (h1, w1, v11), (_h1, w2, v12), (h2, _w1, v21), (_h2, _w2, v22) = corners
    
    if h1 != _h1 or h2 != _h2 or w1 != _w1 or w2 != _w2:
        raise ValueError('points do not form a rectangle')
    if not h1 <= h_origin <= h2 or not w1 <= w_origin <= w2:
        print(h_origin, h1, h2, w_origin, w1, w2)
        raise ValueError('(x, y) not within the rectangle')

    res = (v11 * (h2 - h_origin) * (w2 - w_origin) +
        v21 * (h_origin - h1) * (w2 - w_origin) +
        v12 * (h2 - h_origin) * (w_origin - w1) +
        v22 * (h_origin - h1) * (w_origin - w1)
        ) / ((h2 - h1) * (w2 - w1))
    return res

# corners are 4-by-2 arrays, representing the four image corner (x, y) pairs
def transform(img, canvas, corners):
    """
    :return: canvas after img embeded
    """
    h, w, ch = img.shape

    # TODO: some magic
    ## solve homography matrix
    corners = np.array([[c[1], c[0]] for c in corners])
    src = np.array([[0, 0], [0, w-1], [h-1, 0], [h-1, w-1]])
    H = solve_homography(src, corners)

    ## transformation
    H_inv = np.linalg.inv(H)
    for i in range(min(corners[:, 0]), max(corners[:, 0])+1):
        for j in range(min(corners[:, 1]), max(corners[:, 1])+1):
            coor = np.array([i, j, 1])
            transformed = H_inv.dot(coor)
            transformed /= transformed[2]
            h_trans, w_trans = int(transformed[0]), int(transformed[1])
            if h_trans >= 0 and h_trans <= h-1 and w_trans >= 0 and w_trans <= w-1:
                canvas[i, j] = img[h_trans, w_trans]
    return canvas

def backward_warping(img, transformd_size, dst):
    """
    :return: image after unwarping
    """
    new_h, new_w = transformd_size
    output = np.zeros((new_h, new_w, 3))
    ## solve homography matrix
    dst = np.array([[c[1], c[0]] for c in dst])
    src = np.array([[0, 0], [0, new_w-1], [new_h-1, 0], [new_h-1, new_w-1]])
    H = solve_homography(src, dst)
    for i in range(new_h):
        for j in range(new_w):
            coor = np.array([i, j, 1])
            transformed = H.dot(coor)
            transformed /= transformed[2]
            ## bilinear interpolation
            h, w = int(transformed[0]), int(transformed[1])
            corners = [(h, w, img[h, w]), 
            (h, w+1, img[h, w+1]), 
            (h+1, w, img[h+1, w]), 
            (h+1, w+1, img[h+1, w+1])]
            # print(img[h, w])
            output[i, j] = bilinear_interpolation(transformed[0], transformed[1], corners)
    return output

def viewImage(img):
    cv2.namedWindow("enhanced",0);
    cv2.resizeWindow("enhanced", 640, 480);
    cv2.imshow('enhanced', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    # Part 1
    print('[Part 1...]')
    ts = time.time()
    canvas = cv2.imread('./input/Akihabara.jpg')
    img1 = cv2.imread('./input/lu.jpeg')
    img2 = cv2.imread('./input/kuo.jpg')
    img3 = cv2.imread('./input/haung.jpg')
    img4 = cv2.imread('./input/tsai.jpg')
    img5 = cv2.imread('./input/han.jpg')

    canvas_corners1 = np.array([[779,312],[1014,176],[739,747],[978,639]])
    canvas_corners2 = np.array([[1194,496],[1537,458],[1168,961],[1523,932]])
    canvas_corners3 = np.array([[2693,250],[2886,390],[2754,1344],[2955,1403]])
    canvas_corners4 = np.array([[3563,475],[3882,803],[3614,921],[3921,1158]])
    canvas_corners5 = np.array([[2006,887],[2622,900],[2008,1349],[2640,1357]])

    # TODO: some magic
    canvas = transform(img1, canvas, canvas_corners1)
    canvas = transform(img2, canvas, canvas_corners2)
    canvas = transform(img3, canvas, canvas_corners3)
    canvas = transform(img4, canvas, canvas_corners4)
    canvas = transform(img5, canvas, canvas_corners5)

    cv2.imwrite('part1.png', canvas)
    te = time.time()
    print('Elapse time: {}...'.format(te-ts))


    # Part 2
    print('[Part 2...]')
    ts = time.time()
    img = cv2.imread('./input/QR_code.jpg')

    # TODO: some magic
    dst = np.array([[1985, 1247],[2041, 1219],[2027, 1397],[2084, 1367]])
    h, w = max(dst[:, 1]) - min(dst[:, 1]), max(dst[:, 0]) - min(dst[:, 0])
    output2 = backward_warping(img, (h, w), dst)

    cv2.imwrite('part2.png', output2)
    te = time.time()
    print('Elapse time: {}...'.format(te-ts))


    # Part 3
    print('[Part 3...]')
    # ts = time.time()
    img_front = cv2.imread('./input/crosswalk_front.jpg')

    # TODO: some magic
    dst = np.array([[160, 129],[582, 129],[0, 289],[723, 289]])
    h, w = 400, 500
    output3 = backward_warping(img_front, (h, w), dst)

    cv2.imwrite('part3.png', output3)
    te = time.time()
    print('Elapse time: {}...'.format(te-ts))

if __name__ == '__main__':
    main()
