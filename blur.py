import argparse
import cv2
import yaml
import numpy as np
import sys
sys.path.append('../3DDFA_V2')

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def to_pixel(point:np.ndarray)->np.ndarray:
    """ returning rounds items and return ndarray of integers """
    return np.round(point).astype(int)

def blend(img1, img2):
    """Blend img1 and img2 with img2 alpha layer and put the result in img1.
    img1.shape = h,w,3
    img2.shape = h,w,4
    """
    i1 = img1.astype(np.uint16)
    i2 = img2.astype(np.uint16)
    a2 = i2[:,:,3]
    a1 = 255 - a2
    for i in range(3):
       i1[:,:,i] = (a1*i1[:,:,i]+a2*i2[:,:,i])/255
    img1[:,:,:] = i1.astype(np.uint8)

class TddfaEngine:
    def __init__(self, onnx=True):
        cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)
        if args.onnx:  # Init FaceBoxes and TDDFA, recommend using onnx flag
            import os
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
            os.environ['OMP_NUM_THREADS'] = '4'
            from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
            from TDDFA_ONNX import TDDFA_ONNX
            print("using onnx runtime")
            self.face_boxes = FaceBoxes_ONNX(timer_flag=True)
            self.tddfa = TDDFA_ONNX(**cfg)  # 3D Dense Face Alignment
        else:
            gpu_mode = args.mode == 'gpu'
            self.tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
            self.face_boxes = FaceBoxes(timer_flag=True)

class BlurImage:
    def __init__(self, img, engine:TddfaEngine):
        self.img=img
        self.h, self.w = img.shape[:2]
        self.engine=engine
        self.pts=[] # list of face landmarks
        self.faces = []
        self.mask = np.zeros((self.h, self.w, 3), dtype=np.uint8)

    def process(self):
        """ Detect faces, get 3DMM params and roi boxes """
        tddfa = self.engine.tddfa
        face_boxes = self.engine.face_boxes
        boxes = face_boxes(self.img)
        n = len(boxes)
        print(f'Detect {n} faces')
        if n == 0:
            return
        param_lst, roi_box_lst = tddfa(self.img, boxes)
        print("# params", param_lst[0].shape)
        self.pts = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)  # we want all the face
        if not type(self.pts) in [tuple, list]: # does really happen when only 1 face?
            self.pts = [self.pts]
        print(self.pts[0].shape)

    def draw(self):
        for i in range(len(self.pts)):
            face = Face(self, self.pts[i])
            self.faces.append(face)
            face.process()
            cv2.max( self.mask, face.mask, self.mask)

    def blur(self):
        img_blurred = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        r=0
        for face in self.faces:
            r=max(r,face.radius)
        d=2*((5*r+1)//5)+1 # kernel size has to be odd
        sigma=2*r
        print("max radius", r, "d", d, 'sigma', sigma)
        cv2.GaussianBlur(src=self.img, ksize=(d, d), sigmaX=sigma, sigmaY=sigma, dst=img_blurred)
        blend(self.img, np.dstack((img_blurred, self.mask)))

    def show(self):
        img=self.img
        plt.figure(figsize=(12, self.h / self.w * 12))
        plt.imshow( cv2.cvtColor(img, cv2.COLOR_BGR2RGB) )
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.axis('off')
        plt.show()

class Face:
    """ let's have a custom mask for each face : we want to dilate and blur with kernels adapted for each face geometry """
    def __init__(self, image:BlurImage, landmarks):
        self.image=image
        self.landmarks=landmarks # 38365 points in 3D
        self.radius=10.0 # how to initialize?
        self.mask = np.zeros((self.image.h, self.image.w, 3), dtype=np.uint8)

    def process(self):
        landmarks = self.landmarks
        points = landmarks[[0, 1], ::36].T
        hull = ConvexHull(points)
        vertices = points[hull.vertices]
        vertices = to_pixel(vertices)
        cv2.fillPoly(self.mask, pts=[vertices], color=(255, 255, 255))
        ret, thresh = cv2.threshold(cv2.cvtColor(self.mask, cv2.COLOR_RGB2GRAY), 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, 1, 2)
        cnt = contours[0]
        x, y, w, h = cv2.boundingRect(cnt)
        r = min(h, w) // 10
        self.radius = r
        print("radius", r)
        d = 2*r+1
        cv2.dilate(self.mask, kernel=np.ones((r, r), np.uint8), dst=self.mask)
        cv2.GaussianBlur(src=self.mask, ksize=(d, d), sigmaX=r, sigmaY=r, dst=self.mask)

def main(args):
    image = BlurImage(img = cv2.imread(args.img_fp), engine=TddfaEngine() )
    image.process()
    image.draw()
    image.blur()
    image.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The demo of still image of 3DDFA_V2')
    parser.add_argument('-c', '--config', type=str, default='../3DDFA_V2/configs/mb1_120x120.yml')
    parser.add_argument('-f', '--img_fp', type=str, default='../face-blur/data/students2.jpg')
    parser.add_argument('-m', '--mode', type=str, default='cpu', help='gpu or cpu mode')
    parser.add_argument('--onnx', action='store_true', default=True)
    args = parser.parse_args()
    main(args)
