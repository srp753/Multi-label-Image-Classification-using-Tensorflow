# CNNs-with-Tensorflow
Implementation of simple CNN on MNIST, VGG16 and Alexnet on Pascal VOC dataset

1) 00_mnist.py: Contains code for MNIST 10-digit classification in Tensorflow

2) 01_pascal.py: CNN architecture for MNIST on Pascal VOC dataset

3) 02_pascal_alexnet.py: Alexnet on Pascal VOC

4) 03_pascal_vgg16.py: VGG16 on Pascal VOC from scratch

5) 04_pascal_vggfinetune.py : Fine-tuning VGG16 on Pascal VOC using pre-trained weights

6) 5a_conv1.py: Script to generate conv1 visualisation features. 
	gist_cifar10_train.py : Needed to run 5a_conv1.py

i) Place 5a_conv1.py in the created directory containing the ckpt files(obtained from train).
ii) Run 5a_conv1.py to obtain a folder containing the tensor board object.
iii) Run tensor board —logdir= image_filters
