import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from Functions import *
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split
import pickle
import os.path
from scipy.ndimage.measurements import label

# Read in car and non-car images
noncar_folders = [r'./TrainingData/non-vehicles/GTI', r'./TrainingData/non-vehicles/Extras/']
car_folders = [r'./TrainingData/vehicles/GTI_Far', r'./TrainingData/vehicles/GTI_Left',
               r'./TrainingData/vehicles/GTI_MiddleClose',
               r'./TrainingData/vehicles/GTI_Right', r'./TrainingData/vehicles/KITTI_extracted']

cars = []
notcars = []
for folder in noncar_folders:
    regName = folder + r'/*.png'
    images = glob.glob(regName)
    for image in images:
        notcars.append(image)

for folder in car_folders:
    regName = folder + r'/*.png'
    images = glob.glob(regName)
    for image in images:
        cars.append(image)

# performs under different binning scenarios
color_space = 'YUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 11  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = 0  # Can be 0, 1, 2, or "ALL"

spatial_size = (16, 16)  # Spatial binning dimensions
hist_bins = 8  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off
y_start_stop = [350, 600]  # Min and max in y to search in slide_window()
scale = 2
# dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
# svc = dist_pickle["svc"]
# X_scaler = dist_pickle["scaler"]
# orient = dist_pickle["orient"]
# pix_per_cell = dist_pickle["pix_per_cell"]
# cell_per_block = dist_pickle["cell_per_block"]
# spatial_size = dist_pickle["spatial_size"]
# hist_bins = dist_pickle["hist_bins"]

print('The number of car images is %s' % len(cars))
print('The number of non-car images is %s' % len(notcars))

print('Start to extract the features...')
# Step through the list and search for chessboard corners
plotFigure = False

if plotFigure is True:
    fig, axs = plt.subplots(5, 6, figsize=(8, 8))
    fig.subplots_adjust(hspace=.2, wspace=.001)
    axs = axs.ravel()

    for i in np.arange(15):
        img = cv2.imread(cars[np.random.randint(0, len(cars))])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axs[i].axis('off')
        axs[i].set_title('car', fontsize=10)
        axs[i].imshow(img)
    for i in np.arange(15, 30):
        img = cv2.imread(notcars[np.random.randint(0, len(notcars))])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axs[i].axis('off')
        axs[i].set_title('no-car', fontsize=10)
        axs[i].imshow(img)
    plt.show()

    car_img = mpimg.imread(cars[np.random.randint(0, len(cars))])
    car_img1 = convert_color(car_img, conv='RGB2YUV')
    _, car_dst = get_hog_features(car_img1[:, :, 0], orient=9, pix_per_cell=8, cell_per_block=2, vis=True,
                                  feature_vec=True)
    noncar_img = mpimg.imread(notcars[np.random.randint(0, len(notcars))])
    noncar_img1 = convert_color(noncar_img, conv='RGB2YUV')
    _, noncar_dst = get_hog_features(noncar_img1[:, :, 0], orient=9, pix_per_cell=8, cell_per_block=2, vis=True,
                                     feature_vec=True)

    # Visualize HOG
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(7, 7))
    f.subplots_adjust(hspace=.4, wspace=.2)
    ax1.imshow(car_img)
    ax1.set_title('Car Image', fontsize=16)
    ax2.imshow(car_img1)
    ax2.set_title('Car Image Ch0', fontsize=16)
    ax3.imshow(car_dst, cmap='gray')
    ax3.set_title('Car HOG', fontsize=16)
    ax4.imshow(noncar_img)
    ax4.set_title('Non-Car Image', fontsize=16)
    ax5.imshow(noncar_img1)
    ax5.set_title('Non-Car Ch0', fontsize=16)
    ax6.imshow(noncar_dst, cmap='gray')
    ax6.set_title('Non-Car HOG', fontsize=16)

    plt.show()
#
Feature_Update = False
SVM_Update = False
ImageTest = False
TestingScore = False
if os.path.isfile('features.p') is False or (Feature_Update is True):
    car_features = extract_features(cars, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features))
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    with open("features.p", "wb") as f:
        pickle.dump((scaled_X, y, X_scaler), f)
else:
    with open('features.p', 'rb') as f:
        scaled_X, y, X_scaler = pickle.load(f)

if TestingScore:
    #
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)
    #
    print('Using:', orient, 'orientations', pix_per_cell,
          'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))

if os.path.isfile('SVM_model.p') is False or (SVM_Update is True):
    #
    # # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')

    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t = time.time()
    n_predict = len(X_test)
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these', n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')

    if round(svc.score(X_test, y_test), 4) > 0.97:
        print('Trained SVM model is good enough, save the model')
        SVMModel_Pickle = {}
        SVMModel_Pickle['Model'] = svc
        SVMModel_Pickle['color_space'] = color_space
        SVMModel_Pickle['orient'] = orient
        SVMModel_Pickle['pix_per_cell'] = pix_per_cell
        SVMModel_Pickle['cell_per_block'] = cell_per_block
        SVMModel_Pickle['hog_channel'] = hog_channel
        SVMModel_Pickle['spatial_size'] = spatial_size
        SVMModel_Pickle['hist_bins'] = hist_bins
        SVMModel_Pickle['spatial_feat'] = spatial_feat
        SVMModel_Pickle['hist_feat'] = hist_feat
        SVMModel_Pickle['hog_feat'] = hog_feat
        SVMModel_Pickle['y_start_stop'] = y_start_stop
        SVMModel_Pickle['X_scaler'] = X_scaler

        with open('SVM_model.p', 'wb') as f:
            pickle.dump(SVMModel_Pickle, f)

else:
    with open('SVM_model.p', 'rb') as f:
        SVMModel_Pickle = pickle.load(f)

    print('SVM model has ben loaded.')
    svc = SVMModel_Pickle['Model']
    color_space = SVMModel_Pickle['color_space']
    orient = SVMModel_Pickle['orient']
    pix_per_cell = SVMModel_Pickle['pix_per_cell']
    cell_per_block = SVMModel_Pickle['cell_per_block']
    hog_channel = SVMModel_Pickle['hog_channel']
    spatial_size = SVMModel_Pickle['spatial_size']
    hist_bins = SVMModel_Pickle['hist_bins']
    spatial_feat = SVMModel_Pickle['spatial_feat']
    hist_feat = SVMModel_Pickle['hist_feat']
    hog_feat = SVMModel_Pickle['hog_feat']
    y_start_stop = SVMModel_Pickle['y_start_stop']
    X_scaler = SVMModel_Pickle['X_scaler']
    if TestingScore:
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

if ImageTest is True:
    test_images = glob.glob('./test_images/*jpg')
    for image in test_images:
        img = mpimg.imread(image)

        all_box = []
        for scale in range(0, 10, 3):
            scale = 1. + scale * 0.1
            draw_image, boxes = find_cars(img, ystart=y_start_stop[0], ystop=y_start_stop[1], scale=scale, svc=svc,
                                          X_scaler=X_scaler, orient=orient,
                                          pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                          spatial_size=spatial_size, hist_bins=hist_bins,
                                          cspace=color_space, hog_channel=hog_channel, spatial_feat=spatial_feat,
                                          hist_feat=hist_feat,
                                          hog_feat=True)
            all_box.append(boxes)
        # print(boxes)
        all_box = [box for boxlist in all_box for box in boxlist]
        draw_image = draw_boxes(img, all_box, color=(0, 0, 255), thick=2)
        plt.figure()
        plt.imshow(draw_image)
        folder, save_name = os.path.split(image)
        filename1 = os.path.splitext(save_name)[0]
        filename = './output_images/' + '%s_boxes.png' % filename1
        plt.savefig(filename)

        init_image = np.zeros_like(draw_image[:,:,0])
        init_image = add_heat(init_image, all_box)
        heatmap = np.clip(init_image, 0, 255)
        print(heatmap.max())
        plt.figure()
        plt.imshow(heatmap, cmap='hot')
        filename = './output_images/' + '%s_heatmap.png' % filename1
        plt.savefig(filename)

        detect_img = apply_threshold(init_image, 20)
        plt.figure()
        plt.imshow(detect_img, cmap='hot')
        filename = './output_images/' + '%s_filtered.png' % filename1
        plt.savefig(filename)

        labels = label(detect_img)
        print(labels[0])
        plt.figure()
        plt.imshow(labels[0], cmap='gray')
        filename = './output_images/' + '%s_labelled.png' % filename1
        plt.savefig(filename)
        print(labels[1], 'cars found')

        draw_img = draw_labeled_bboxes(np.copy(img), labels)
        # Display the image
        plt.figure()
        plt.imshow(draw_img)
        filename = './output_images/' + '%s_Final.png' % filename1
        plt.savefig(filename)

videoCaptureImageTest = False
if videoCaptureImageTest is True:
    test_images = glob.glob('./VideoCapture/*.jpg')
    for image in test_images:
        img = mpimg.imread(image)

        all_box = []
        for scale in range(0, 10, 3):
            scale = 1. + scale * 0.1
            draw_image, boxes = find_cars(img, ystart=y_start_stop[0], ystop=y_start_stop[1], scale=scale, svc=svc,
                                          X_scaler=X_scaler, orient=orient,
                                          pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                          spatial_size=spatial_size, hist_bins=hist_bins,
                                          cspace=color_space, hog_channel=hog_channel, spatial_feat=spatial_feat,
                                          hist_feat=hist_feat,
                                          hog_feat=True)
            all_box.append(boxes)
        # print(boxes)
        all_box = [box for boxlist in all_box for box in boxlist]
        draw_image = draw_boxes(img, all_box, color=(0, 0, 255), thick=2)
        plt.figure()
        plt.imshow(draw_image)
        folder, save_name = os.path.split(image)
        filename1 = os.path.splitext(save_name)[0]
        filename = './VideoCaptureOutput/' + '%s_boxes.png' % filename1
        plt.savefig(filename)

        init_image = np.zeros_like(draw_image[:, :, 0])
        init_image = add_heat(init_image, all_box)
        heatmap = np.clip(init_image, 0, 255)
        print(heatmap.max())
        plt.figure()
        plt.imshow(heatmap, cmap='hot')
        filename = './VideoCaptureOutput/' + '%s_heatmap.png' % filename1
        plt.savefig(filename)

        detect_img = apply_threshold(init_image, 20)
        plt.figure()
        plt.imshow(detect_img, cmap='hot')
        filename = './VideoCaptureOutput/' + '%s_filtered.png' % filename1
        plt.savefig(filename)

        labels = label(detect_img)
        print(labels[0])
        plt.figure()
        plt.imshow(labels[0], cmap='gray')
        filename = './VideoCaptureOutput/' + '%s_labelled.png' % filename1
        plt.savefig(filename)
        print(labels[1], 'cars found')

        draw_img = draw_labeled_bboxes(np.copy(img), labels)
        # Display the image
        plt.figure()
        plt.imshow(draw_img)
        filename = './VideoCaptureOutput/' + '%s_Final.png' % filename1
        plt.savefig(filename)

        f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(50, 10))
        ax1.imshow(draw_image)
        ax1.set_title('Boxs',fontsize=50)
        ax2.imshow(heatmap, cmap='hot')
        ax2.set_title('HeatMap',fontsize=50)
        ax3.imshow(detect_img, cmap='hot')
        ax3.set_title('Filtered HeatMap',fontsize=50)
        ax4.imshow(labels[0], cmap='gray')
        ax4.set_title('Label',fontsize=50)
        ax5.imshow(draw_img)
        ax5.set_title('Final Detection Image',fontsize=50)
        filename = './VideoCaptureOutput/' + '%s_Combined.png' % filename1
        plt.savefig(filename,bbox_inches='tight')

def video_process(img):
    all_box = []
    global previous_labels
    global frameNum
    global threshold_set
    img_write = False
    if img_write:
        cv2.imwrite("./VideoCapture/frame%d.jpg" % frameNum, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # scale = 0.25
    # draw_image, boxes = find_cars(img, ystart=400, ystop=450, scale=scale, svc=svc,
    #                               X_scaler=X_scaler, orient=orient,
    #                               pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
    #                               spatial_size=spatial_size, hist_bins=hist_bins,
    #                               cspace=color_space, hog_channel=hog_channel, spatial_feat=spatial_feat,
    #                               hist_feat=hist_feat,
    #                               hog_feat=True)
    # all_box.append(boxes)
    for scale in range(0, 10, 3):
        scale = 1 + scale * 0.1
        draw_image, boxes = find_cars(img, ystart=y_start_stop[0], ystop=y_start_stop[1], scale=scale, svc=svc,
                                      X_scaler=X_scaler, orient=orient,
                                      pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                      spatial_size=spatial_size, hist_bins=hist_bins,
                                      cspace=color_space, hog_channel=hog_channel, spatial_feat=spatial_feat,
                                      hist_feat=hist_feat,
                                      hog_feat=True)
        all_box.append(boxes)
    # print(boxes)
    all_box = [box for boxlist in all_box for box in boxlist]
    init_image = np.zeros_like(img[:,:,0])
    init_image = add_heat(init_image, all_box)
    detect_img = apply_threshold(init_image, threshold_set, previous_labels)
    labels = label(detect_img)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    previous_labels = labels


    frameNum +=1
    return draw_img

def video_process_MultiFrame(img):
    global all_box
    global previous_labels
    global frameNum
    global threshold_set
    FrameSetNum = 4
    CurrentMode = frameNum % FrameSetNum

    img_write = False
    if img_write:
        cv2.imwrite("./VideoCapture/frame%d.jpg" % frameNum, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # scale = 0.25
    # draw_image, boxes = find_cars(img, ystart=400, ystop=450, scale=scale, svc=svc,
    #                               X_scaler=X_scaler, orient=orient,
    #                               pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
    #                               spatial_size=spatial_size, hist_bins=hist_bins,
    #                               cspace=color_space, hog_channel=hog_channel, spatial_feat=spatial_feat,
    #                               hist_feat=hist_feat,
    #                               hog_feat=True)
    # all_box.append(boxes)

    scale = 1 + CurrentMode * 0.3
    draw_image, boxes = find_cars(img, ystart=y_start_stop[0], ystop=y_start_stop[1], scale=scale, svc=svc,
                                  X_scaler=X_scaler, orient=orient,
                                  pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                  spatial_size=spatial_size, hist_bins=hist_bins,
                                  cspace=color_space, hog_channel=hog_channel, spatial_feat=spatial_feat,
                                  hist_feat=hist_feat,
                                  hog_feat=True)
    if len(all_box) >= FrameSetNum:
        all_box.pop(0)
    all_box.append(boxes)
    # print(boxes)

    all_box_list = [box for boxlist in all_box for box in boxlist]
    init_image = np.zeros_like(img[:,:,0])
    init_image = add_heat(init_image, all_box_list)
    detect_img = apply_threshold(init_image, threshold_set, previous_labels)
    labels = label(detect_img)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    previous_labels = labels

    result_check = True
    if result_check:
        draw_image = draw_boxes(img, all_box_list, color=(0, 0, 255), thick=2)
        f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(50, 10))
        ax1.imshow(draw_image)
        ax1.set_title('Boxs', fontsize=50)
        heatmap = np.clip(init_image, 0, 255)
        ax2.imshow(heatmap, cmap='hot')
        ax2.set_title('HeatMap', fontsize=50)
        ax3.imshow(detect_img, cmap='hot')
        ax3.set_title('Filtered HeatMap', fontsize=50)
        ax4.imshow(labels[0], cmap='gray')
        ax4.set_title('Label', fontsize=50)
        ax5.imshow(draw_img)
        ax5.set_title('Final Detection Image', fontsize=50)
        filename = './VideoCaptureOutput/' + 'MultiFrame%s_Combined_TH%s.png' % (frameNum,threshold_set)
        plt.savefig(filename, bbox_inches='tight')



    frameNum +=1
    return draw_img

from moviepy.editor import VideoFileClip

test_video = False
#
frameNum = 0
if test_video:
    previous_labels = None
    test_out_file = 'test_video_out.mp4'
    clip_test = VideoFileClip('test_video.mp4')
    clip_test_out = clip_test.fl_image(video_process_MultiFrame)
    clip_test_out.write_videofile(test_out_file, audio=False)


project_video = True

if project_video:
    # TH_List = list(range(20,50,5))
    TH_List = [30]
    for i in TH_List:
        threshold_set = i
        previous_labels = None
        frameNum = 0
        all_box = []
        project_out_file = 'project_video_out_multiFrame_TH%s.mp4' %threshold_set
        clip_test = VideoFileClip('project_video_multiFrame.mp4')
        clip_test_out = clip_test.fl_image(video_process_MultiFrame)
        clip_test_out.write_videofile(project_out_file, audio=False)



"""
Make sure your images are scaled correctly
The training dataset provided for this project ( vehicle and non-vehicle images) are in the .png format. 
Somewhat confusingly, matplotlib image will read these in on a scale of 0 to 1, but cv2.imread() will scale them from 0 to 255. 
Be sure if you are switching between cv2.imread() and matplotlib image for reading images that you scale them appropriately! 
Otherwise your feature vectors can get screwed up.

To add to the confusion, matplotlib image will read .jpg images in on a scale of 0 to 255 so if you are testing your pipeline on .jpg images remember to scale them accordingly. 
And if you take an image that is scaled from 0 to 1 and change color spaces using cv2.cvtColor() you'll get back an image scaled from 0 to 255. 
So just be sure to be consistent between your training data features and inference features!
"""
