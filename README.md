# Computer-Vision-Projects
Two different works:
1) The first includes an SVM classifier for 101 classes.
-	The data contains 101 classes, with 31-800 images each.
- The classifier is an SVM + HOG pipe.

2) The second project is a flower classifier, which detects on whic images there appear a flower or not.
The images were collected from the Israeli Volkani institute, and includes 472 cropped images of flowers and non-flowers, with corresponding labels.
We built a flower classifier with task transfer from a pre-trained network (ResNet50V2), and improved it with data augmentation in training, weight decay, and architecture changes.
