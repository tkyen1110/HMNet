rm /home/tkyen/opencv_practice/HMNet_pth/experiments/detection/data/gen1/list
rm /home/tkyen/opencv_practice/HMNet_pth/experiments/detection/data/gen1/train_evt
rm /home/tkyen/opencv_practice/HMNet_pth/experiments/detection/data/gen1/train_lbl
rm /home/tkyen/opencv_practice/HMNet_pth/experiments/detection/data/gen1/val_evt
rm /home/tkyen/opencv_practice/HMNet_pth/experiments/detection/data/gen1/val_lbl
rm /home/tkyen/opencv_practice/HMNet_pth/experiments/detection/data/gen1/test_evt
rm /home/tkyen/opencv_practice/HMNet_pth/experiments/detection/data/gen1/test_lbl

ln -s /home/tkyen/opencv_practice/data_1/Gen1_Automotive/HMNet_Gen4to1/list /home/tkyen/opencv_practice/HMNet_pth/experiments/detection/data/gen1/list
# ln -s /home/tkyen/opencv_practice/data_1/Gen1_Automotive/HMNet_Gen4to1/train_evt /home/tkyen/opencv_practice/HMNet_pth/experiments/detection/data/gen1/train_evt
# ln -s /home/tkyen/opencv_practice/data_1/Gen1_Automotive/HMNet_Gen4to1/train_lbl /home/tkyen/opencv_practice/HMNet_pth/experiments/detection/data/gen1/train_lbl
# ln -s /home/tkyen/opencv_practice/data_1/Gen1_Automotive/HMNet_Gen4to1/val_evt /home/tkyen/opencv_practice/HMNet_pth/experiments/detection/data/gen1/val_evt
# ln -s /home/tkyen/opencv_practice/data_1/Gen1_Automotive/HMNet_Gen4to1/val_lbl /home/tkyen/opencv_practice/HMNet_pth/experiments/detection/data/gen1/val_lbl
ln -s /home/tkyen/opencv_practice/data_1/Gen1_Automotive/HMNet_Gen4to1/test_evt /home/tkyen/opencv_practice/HMNet_pth/experiments/detection/data/gen1/test_evt
ln -s /home/tkyen/opencv_practice/data_1/Gen1_Automotive/HMNet_Gen4to1/test_lbl /home/tkyen/opencv_practice/HMNet_pth/experiments/detection/data/gen1/test_lbl