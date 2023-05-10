import os
import cv2
import dlib
from imutils import face_utils
import pandas as pd
import time
from PIL import Image, ImageOps
import numpy as np
import torch
from torchvision import transforms
from face_detection import RetinaFace
from models.gaze.gazenet import get_model
import torch.nn as nn
import torch.backends.cudnn as cudnn
from model import gaze_network

from utils import select_device

from head_pose import HeadPoseEstimator

trans = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Resize((640,640)),  # this line just in case of hrnet 640
        transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def estimateHeadPose(landmarks, face_model, camera, distortion, iterate=True):
    ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, flags=cv2.SOLVEPNP_EPNP)

    ## further optimize
    if iterate:
        ret, rvec, tvec = cv2.solvePnP(face_model, landmarks, camera, distortion, rvec, tvec, True)

    return rvec, tvec

def draw_gaze(image_in, pitchyaw, thickness=2, color=(0, 0, 255)):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    (h, w) = image_in.shape[:2]
    length = w / 2.0
    pos = (int(h / 2.0), int(w / 2.0))
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1]) * np.cos(pitchyaw[0])
    dy = -length * np.sin(pitchyaw[0])
    cv2.arrowedLine(image_out, tuple(np.round(pos).astype(np.int32)),
                   tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
                   thickness, cv2.LINE_AA, tipLength=0.2)
    return image_out

def normalizeData_face(img, face_model, landmarks, hr, ht, cam):
    ## normalized camera parameters
    focal_norm = 960  # 960  # focal length of normalized camera
    distance_norm = 600  # normalized distance between eye and camera (600 in the original code and 300 in the paper)
    roiSize = (224, 224)  # size of cropped eye image (in the original code)
    # roiSize = (448, 448)  # size of cropped eye image (in the paper)
    # roiSize = (640, 640) # same value of the required input features

    ## compute estimated 3D positions of the landmarks
    ht = ht.reshape((3, 1))
    hR = cv2.Rodrigues(hr)[0]  # rotation matrix
    Fc = np.dot(hR, face_model.T) + ht  # rotate and translate the face model
    two_eye_center = np.mean(Fc[:, 0:4], axis=1).reshape((3, 1))
    nose_center = np.mean(Fc[:, 4:6], axis=1).reshape((3, 1))
    # get the face center
    face_center = np.mean(np.concatenate((two_eye_center, nose_center), axis=1), axis=1).reshape((3, 1))

    ## ---------- normalize image ----------
    distance = np.linalg.norm(face_center)  # actual distance between eye and original camera
    # print(distance)
    z_scale = distance_norm / distance
    cam_norm = np.array([  # camera intrinsic parameters of the virtual camera
        [focal_norm, 0, roiSize[0] / 2],
        [0, focal_norm, roiSize[1] / 2],
        [0, 0, 1.0],
    ])
    S = np.array([  # scaling matrix
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, z_scale],
    ])

    hRx = hR[:, 0]
    forward = (face_center / distance).reshape(3)
    down = np.cross(forward, hRx)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    R = np.c_[right, down, forward].T  # rotation matrix R

    W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam)))  # transformation matrix

    img_warped = cv2.warpPerspective(img, W, roiSize)  # warp the input image

    # head pose after normalization
    hR_norm = np.dot(R, hR)  # head pose rotation matrix in normalized space
    hr_norm = cv2.Rodrigues(hR_norm)[0]  # convert rotation matrix to rotation vectors

    # normalize the facial landmarks
    num_point = landmarks.shape[0]
    landmarks_warped = cv2.perspectiveTransform(landmarks, W)
    landmarks_warped = landmarks_warped.reshape(num_point, 2)

    return img_warped, landmarks_warped

if __name__ == '__main__':
    start_time = time.time()
    cudnn.enabled = True
    batch_size = 24
    gpu = select_device('0', batch_size=batch_size)
    # processing from local
    image_folder = './input/4k_exp_setting'
    output_folder = './output/4k'
    # video_name = '.processed_video.avi'

    # torch.PYTORCH_CUDA_ALLOC_CONF('max_split_size_mb':10)
    # img_file_name = './example/input/frame586.jpg'
    # print('load input face image: ', img_file_name)
    # image = cv2.imread(img_file_name)

    # processing with local images
    # images = [int(img[5:-4]) for img in os.listdir(image_folder) if img.endswith(".jpg")] #remove 'frame' and order images
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]  # from pepper experiment
    images = sorted(images)
    # print("images[0]:", images[0])
    # frame = cv2.imread(os.path.join(image_folder, "frame" + str(images[0]) + ".jpg"))
    frame = cv2.imread(os.path.join(image_folder, str(images[0])))

    predictor = dlib.shape_predictor('./modules/shape_predictor_68_face_landmarks.dat')
    # face_detector = dlib.cnn_face_detection_model_v1('./modules/mmod_human_face_detector.dat')
    # face_detector = dlib.get_frontal_face_detector()  ## this face detector is not very powerful
    face_detector = RetinaFace()

    # load camera information
    cam_file_name = './example/input/webcam.xml'  #[cam_pepper_10.xml] this is camera calibration information file obtained with OpenCV for low resolution. [webcam.xml]for 4k camera
    if not os.path.isfile(cam_file_name):
        print('no camera calibration file is found.')
        exit(0)

    fs = cv2.FileStorage(cam_file_name, cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode('Camera_Matrix').mat()  # camera calibration information is used for data normalization
    camera_distortion = fs.getNode('Distortion_Coefficients').mat()
    # print(camera_matrix, camera_distortion)
    # exit(0)
    # load face model
    face_model_load = np.loadtxt('face_model.txt')  # Generic face model with 3D facial landmarks
    landmark_use = [20, 23, 26, 29, 15, 19]  # we use eye corners and nose conners
    face_model = face_model_load[landmark_use, :]
    # print()
    # print('load gaze estimator')
    model = gaze_network() # for baseline
    # model = get_model(name='hrnet_w64')  # hrnet_w64, botnet

    model.cuda(gpu)  # comment this line out if you are not using GPU
    model = nn.DataParallel(model)  # State of art
    # pre_trained_model_path = './ckpt/epoch_12_HRnet.pth.tar'
    # pre_trained_model_path = './ckpt/epoch_26_botnet.pth.tar'
    pre_trained_model_path = './ckpt/epoch_24_ckpt_baseline.pth.tar'

    if not os.path.isfile(pre_trained_model_path):
        print('the pre-trained gaze estimation model does not exist.')
        exit(0)
    else:
        print('load the pre-trained model: ', pre_trained_model_path)
    # ckpt = torch.load(pre_trained_model_path)
    # print(ckpt.keys())
    # model.load_state_dict(ckpt['model_state'], strict=True)  # load the pre-trained model baseline

    checkpoint = torch.load(pre_trained_model_path)


    # from collections import OrderedDict
    #
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[13:]  # remove 'module.' of dataparallel
    #     new_state_dict[name] = v

    # model.module.load_state_dict(checkpoint['state_dict'], strict=True) # HRnet
    model.module.load_state_dict(checkpoint['model_state'], strict=True)  # baseline

    model.eval()  # change it to the evaluation mode

    count = 1
    # we save pitch and yaw into a pandas DataFrame
    pitch_predicted_ = []
    yaw_predicted_ = []
    with torch.no_grad():
        for image in images:
            # image = cv2.imread(os.path.join(image_folder, "frame" + str(image) + ".jpg"))
            image = cv2.imread(os.path.join(image_folder, str(image)))
            # detected_faces = face_detector(image, 1)  # baseline face detectors
            detected_faces = face_detector(image)  # Retina face detectron
            # print(detected_faces)
            # if len(detected_faces) == 0:
            #     print('warning: no detected face')
            #     continue
            # print('detected one face')


            detected_face = dlib.rectangle(left=int(detected_faces[0][0][0]), top=int(detected_faces[0][0][1]),\
                            right=int(detected_faces[0][0][2]), bottom=int(detected_faces[0][0][3]))  # for Retina face detection
            # detected_face = detected_faces[0]  # for frontal face detection
            # detected_face = detected_faces[0].rect  # for cnn face detectron

            shape = predictor(image, detected_face) ## only use the first detected face (assume that each input image only contains one face)
            shape = face_utils.shape_to_np(shape)
            # print(shape.shape)
            landmarks = []
            for (x, y) in shape:
                landmarks.append((x, y))
            landmarks = np.asarray(landmarks)
            # print('estimate head pose')
            # estimate the head pose,
            # the complex way to get head pose information, eos library is required,  probably more accurrated
            # landmarks = landmarks.reshape(-1, 2)
            # head_pose_estimator = HeadPoseEstimator()
            # hr, ht, o_l, o_r, _ = head_pose_estimator(image, landmarks, camera_matrix)  # camera_matrix[cam_id]

            # the easy way to get head pose information, fast and simple
            facePts = face_model.reshape(6, 1, 3)
            # print('facePts', facePts)
            landmarks_sub = landmarks[[36, 39, 42, 45, 31, 35], :]
            landmarks_sub = landmarks_sub.astype(float)  # input to solvePnP function must be float type
            landmarks_sub = landmarks_sub.reshape(6, 1, 2)  # input to solvePnP requires such shape

            hr, ht = estimateHeadPose(landmarks_sub, facePts, camera_matrix, camera_distortion)
            # hr, ht = estimateHeadPose(landmarks_sub, facePts, camera_matrix, camera_distortion)

            # data normalization method
            # print('data normalization, i.e. crop the face image')
            img_normalized, landmarks_normalized = normalizeData_face(image, face_model, landmarks_sub, hr, ht, camera_matrix)
            # print("img_normalized",img_normalized.shape)
            # output_path = output_folder + '/frame_2' + '.jpg'
            # cv2.imwrite(output_path, img_normalized)
            # output_path = output_folder + '/frame_3' + '.jpg'
            # cv2.imwrite(output_path, image)


            # prediction
            input_var = img_normalized[:, :, [2, 1, 0]]  # from BGR to RGB
            input_var = trans(input_var)
            # print("input_var", input_var.shape)
            input_var = torch.autograd.Variable(input_var.float().cuda(gpu))
            # input_var = torch.autograd.Variable(input_var.float())
            input_var = input_var.view(1, input_var.size(0), input_var.size(1), input_var.size(2))  # the input must be 4-dimension
            # print(input_var.shape)
            pred_gaze = model(input_var)  # get the output gaze direction, this is 2D output as pitch and raw rotation

            pred_gaze = pred_gaze[0] # here we assume there is only one face inside the image, then the first one is the prediction
            pred_gaze_np = pred_gaze.cpu().data.numpy()  # convert the pytorch tensor to numpy array
            pred_gaze_numpy = pred_gaze.cpu().detach().numpy()
            # print(pred_gaze_numpy)
            pitch_predicted_.append(pred_gaze_numpy[1])
            yaw_predicted_.append(pred_gaze_numpy[0])
            # print(pred_gaze_np, pred_gaze_numpy)
            # print('prepare the output')
            # draw the facial landmarks

            landmarks_normalized = landmarks_normalized.astype(int) # landmarks after data normalization
            for (x, y) in landmarks_normalized:
                cv2.circle(img_normalized, (x, y), 5, (0, 255, 0), -1)
            # print(img_normalized.shape)

            face_patch_gaze = draw_gaze(img_normalized, pred_gaze_np)  # draw gaze direction on the normalized face image
            output_path = output_folder+'/frame_'+str(count)+'.jpg'
            # print('save output image to: ', output_path)
            # print(output_path, face_patch_gaze)
            cv2.imwrite(output_path, face_patch_gaze)
            # exit()
            count +=1
    dataframe = pd.DataFrame(
        data=np.concatenate([np.array(yaw_predicted_, ndmin=2), np.array(pitch_predicted_, ndmin=2)]).T,
        columns=["pitch", "yaw"])
    dataframe.to_csv(os.path.join(output_folder,'pitch_yaw.csv'), index=False)
    print("--- Complete excecution = %s seconds ---" % (time.time() - start_time))