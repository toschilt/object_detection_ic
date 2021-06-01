import cv2

# read image

for i in range(50, 801):
    file_name = 'scene00'
    base_path = '/home/ltoschi/Documents/LabRoM/2021/object_detection_ic/myDatasets/Training/JPEGImages/scene00'
    if(i < 100):
        base_path += '0'
        file_name += '0'

    file_name += str(i)
    img = cv2.imread(base_path + str(i) + '.png', cv2.IMREAD_UNCHANGED)
    
    # get dimensions of image
    dimensions = img.shape
    imgType = img.dtype
    
    # height, width, number of channels in image
    height = img.shape[0]
    width = img.shape[1]
    channels = img.shape[2]
    
    print('File Name          : ',file_name)
    print('File Type          : ',imgType)
    print('Image Dimension    : ',dimensions)
    print('Image Height       : ',height)
    print('Image Width        : ',width)
    print('Number of Channels : ',channels)

for i in range(801, 1051):
    file_name = 'scene00'
    base_path = '/home/ltoschi/Documents/LabRoM/2021/object_detection_ic/myDatasets/Testing/JPEGImages/scene0'
    if(i < 1000):
        base_path += '0'
        file_name += '0'

    file_name += str(i)
    img = cv2.imread(base_path + str(i) + '.png', cv2.IMREAD_UNCHANGED)
    
    # get dimensions of image
    dimensions = img.shape
    imgType = img.dtype
    
    # height, width, number of channels in image
    height = img.shape[0]
    width = img.shape[1]
    channels = img.shape[2]
    
    print('File Name          : ',file_name)
    print('File Type          : ',imgType)
    print('Image Dimension    : ',dimensions)
    print('Image Height       : ',height)
    print('Image Width        : ',width)
    print('Number of Channels : ',channels)