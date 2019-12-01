import numpy as np
import cv2
import sys, time

def main(ref_image, template, video):
    ref_image = cv2.imread(ref_image)  ## load gray if you need.
    template = cv2.imread(template)  ## load gray if you need.
    video = cv2.VideoCapture(video)
    film_h, film_w = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    film_fps = video.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    videowriter = cv2.VideoWriter("ar_video.mp4", fourcc, film_fps, (film_w, film_h))

    # shrink images
    height, width = template.shape[:2]
    template = cv2.resize(template, (800, 800))
    ref_image = cv2.resize(ref_image, (800, 800))

    # create sift feature detector and compute template features
    sift = cv2.xfeatures2d.SIFT_create()
    template_keypoints, template_descriptors = sift.detectAndCompute(template, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    frame_number = 0
    while(video.isOpened()):
        ret, frame = video.read()
        print('Processing frame {}'.format(frame_number))
        if ret:  ## check whethere the frame is legal, i.e., there still exists a frame

            ## TODO: homography transform, feature detection, ransanc, etc.
            # compute frame features
            frame_keypoints, frame_descriptors = sift.detectAndCompute(frame, None)

            # find matching descriptors
            matches = flann.knnMatch(template_descriptors, frame_descriptors, k=2)
            # find good matches
            good = []
            for m,n in matches:
                if m.distance < 0.7 * n.distance:
                    good.append(m)

            # check if match > min count of match features
            if len(good) > 10:
                template_pts = np.float32([ template_keypoints[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                frame_pts = np.float32([ frame_keypoints[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

                homography, mask = cv2.findHomography(template_pts, frame_pts, cv2.RANSAC, 5.0)
                if homography is None:
                    continue

                # project image in video
                h, w, ch = ref_image.shape
                
                for y in range(h):
                    for x in range(w):
                        coor = np.array([x, y, 1])
                        transformed = homography.dot(coor)
                        x_trans, y_trans = int(transformed[0]/transformed[2]), int(transformed[1]/transformed[2])
                        frame[y_trans, x_trans] = ref_image[y, x]

            videowriter.write(frame)
            frame_number += 1
        else:
            break
            
    video.release()
    videowriter.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    ## you should not change this part
    ref_path = './input/sychien.jpg'
    template_path = './input/marker.png'
    video_path = sys.argv[1]  ## path to ar_marker.mp4
    ts = time.time()
    main(ref_path,template_path,video_path)
    te = time.time()
    print('Elapse time: {}...'.format(te-ts))
