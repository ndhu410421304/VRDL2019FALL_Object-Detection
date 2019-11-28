import os
import pandas as pd
import h5py
import cv2

train_folder = "train"

# parameters
img_widths =[]
img_heights = []
img_name = []
labels = []
xmin = []
xmax = []
ymin = []
ymax = []

cur_name = '' # current name

boxcount = 0

trainfile = open("train.txt","w")
valfile = open("val.txt","w")
cur_file = trainfile # current opened file
process_count = 0 # count the image had beend process

# generate path to h5 file
def get_name(index, hdf5_data):
    name = hdf5_data['/digitStruct/name']
    return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]].value])

# Extract bounding box
def get_bbox(index, hdf5_data):
    # 80% for train
    global trainfile
    global valfile
    global cur_file
    global process_count
    if(process_count < 26756): # complete extract from training dataset
        cur_file = trainfile
        cur_file.write("%s data/train/" % process_count)
    else:
        cur_file = valfile
        cur_file.write("%s data/val/" % str(process_count - 26756))
    
    global boxcount # annouce as global
    global cur_name # annouce as global

    cur_file.write(cur_name + " ")

    attrs = {}
    item = hdf5_data['digitStruct']['bbox'][index].item()
    keynum = 0
    # load image to get height, width
    tmp_height, tmp_width = cv2.imread(os.path.join(train_folder,cur_name)).shape[:2]

    cur_file.write(str(tmp_width) + " " + str(tmp_height) + " ")

    '''
    Notice not to let bounding box range
    not to larger than image size
    '''

    for key in ['label', 'left', 'top', 'width', 'height']:
        attr = hdf5_data[item][key]
        values = [hdf5_data[attr.value[i].item()].value[0][0]
                  for i in range(len(attr))] if len(attr) > 1 else [attr.value[0][0]]
        print(values)
        print(len(values))
        print(keynum)
        # keynum from label to height, 0~5 correspond
        for i in range(len(values)):
            print(values[i])
            if keynum == 0:
                labels.append(values[i])

            # normalize range is from 0 to 1
            # so not to let max value greater than 1
            elif keynum == 1:
                xmin.append(max(values[i], 1))

            elif keynum == 2:
                ymin.append(max(values[i], 1))

            elif keynum == 3:
                xmax.append(min(values[i]+xmin[boxcount+i], tmp_width - 1))
                print(values[i]+xmin[boxcount + i])

            # For last keynum, we would construct the data format we are going to
            # use for the model
            elif keynum == 4:
                ymax.append(min(ymin[boxcount + i] + values[i], tmp_height - 1))
                # chnage 0 to 10: for yolo model
                # add spacing for the model required file format
                if(int(labels[boxcount + i]) == 10):
                    cur_file.write(str("0" + " "))
                else:
                    cur_file.write(str(int((labels[boxcount + i]))) + " ")
                cur_file.write(str(int(xmin[boxcount + i])) + " ")
                cur_file.write(str(int(ymin[boxcount + i])) + " ")
                cur_file.write(str(int(xmax[boxcount + i])) + " ")
                cur_file.write(str(int(ymax[boxcount + i])) + " ")
                # append current img name as well
                img_name.append(cur_name)
                print(cur_name)
                if i == (len(values) - 1):
                    boxcount += len(values)
                    # read image size
                    img_h, img_w = tmp_height, tmp_width
                    print(img_h, img_w)
                    for j in range(len(values)):
                        img_widths.append(img_w)
                        img_heights.append(img_h)
        keynum += 1
    process_count += 1
    cur_file.write("\n")

def img_boundingbox_data_constructor(mat_file):
    f = h5py.File(mat_file,'r') 
    print('image bounding box data construction starting...')
    for j in range(f['/digitStruct/bbox'].shape[0]):
        print(j)
        img_names = get_name(j, f)
        print(img_names)
        global cur_name # load current image to global variable
        cur_name = img_names
        get_bbox(j, f)

def construct_all_data(img_folder,mat_file_name):
    img_boundingbox_data_constructor(os.path.join(img_folder,mat_file_name))

construct_all_data(train_folder,'digitStruct.mat')

# close after exporting file
trainfile.close()
valfile.close()

output = pd.DataFrame({"filename": img_name, "width": img_widths, "height": img_heights, "class": labels, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})
output.to_csv("boxrecord.csv", columns=["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"], index=False) #output result to csv
