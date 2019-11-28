## VRDL2019FALL_Object-Detection
To use these project file, you would need to install tensorflow-gpu version 1.x with correspond cuda and cudnn version
Notice that the big model weights, h5 and model file/image folder were removed from the project.You would need to findyou own file to restore them in their correspond location. 

## Tensorflow1
Project file of object recognition program base on the tensorflow api and edjieelectronic's tensorflow object recognition tutorials.
Remeber to set the python path to model, model/researtch and model/research/slim to been able to correctly run the program.
To use this project, you may need to install and do some setting, which can follow instructioon[3] to setup (remeber to download tensorflow api same as your installed version). Once finish setup, ypu can put the train/test file with their correspond annotation csv to image foloder, use generate tfrecord.py to generate the annotation file the model is going to use. Download the desired pretrain model from modelzoo (link can be found at tensorflow1/models/research/object_detection/g3doc/detection_model_zoo.md, where the same folder contain other official guides would need to check when you are going to train your own model), extracrt the model under tensorflow1/models/research/object_detection/, amd found the correspond config file under tensorflow1/models/research/object_detection/samples(if you can't find one, check if your API version is correct.), You may need to adjust the image size, path(and others of your choice), then located it under tensorflow1/models/research/object_detection/training folder. You may need to generate a label.pbtxt under the same folder, with the condition of your training file written in.
After these preparation, you can start training. By train.py and generate result file from Object_detection_image.py, with correct parser command(can be found in instruction[3]). Note that the project file may already contain some cache and logs, which I would strongly recommend to follw the guide to install and operate step by step, and check here to see how the correct implement file would be like.  
## Reference:
# [1]https://github.com/tensorflow/models/tree/d530ac540b0103caa194b4824af353f1b073553b (TF1.9)
# [2]https://github.com/tensorflow/models/blob/d530ac540b0103caa194b4824af353f1b073553b/research/object_detection/g3doc/detection_model_zoo.md (fasterv2 model)
# [3]https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10 (TF inplement)
# [4]https://github.com/wizyoung/YOLOv3_TensorFlow (Yolov3 implement)

d 
