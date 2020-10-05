'''
COMMON: common settings
Written by sunzhu on Sep 29, 2020, version 0.0
'''
import os

# 1. color for displaying
COLORS = [[31, 0, 255], [0, 159, 255], [255, 95, 0], [255, 19, 0],
    [255, 0, 0], [255, 38, 0], [0, 255, 25], [255, 0, 133],
    [255, 172, 0], [108, 0, 255], [0, 82, 255], [0, 255, 6],
    [255, 0, 152], [223, 0, 255], [12, 0, 255], [0, 255, 178]]

# 2. video settings
FRAME_RATE = 25

# 3. Project settings (Default)
ROOT = os.path.dirname(__file__)                        # project path

SRC_IMAGES = os.path.join(ROOT, "data\\images")
SRC_VIDEOS = os.path.join(ROOT, "data\\videos")

DET_RESULT = os.path.join(ROOT, "results\\detection")    # detection results path
ROI_RESULT = os.path.join(ROOT, "results\\roi")          # ROI settings path
SCT_RESULT = os.path.join(ROOT, "results\\sct")          # single camera tracking results
MCT_RESULT = os.path.join(ROOT, "results\\mct")          # multiple cameras tracking results

# 4. Default Test Case settings, ignore these settings in your own project
# cam names
cam_names = ["001", "002", "003", "004", "005"]
# images
data_path = [os.path.join(SRC_IMAGES, cam_names[i]) for i in range(5)]
# bbox information
box_info = [os.path.join(DET_RESULT, cam_names[i] + '\\boxInfo.csv') for i in range(5)]
# MonitoringRegion
roi_info = [os.path.join(ROI_RESULT, cam_names[i] + '\\0001_transformer.json') for i in range(5)]
# Save path
save_path = [os.path.join(SCT_RESULT, cam_names[i]) for i in range(5)]
# tracking information
track_info = [os.path.join(SCT_RESULT, cam_names[i] + '\\tracking_info.json') for i in range(5)]
# associate information
associate_info = os.path.join(MCT_RESULT, 'mct_results.json')
