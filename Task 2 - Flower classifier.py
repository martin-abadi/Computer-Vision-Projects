import os
import cv2
import scipy.io as sio
import scipy as s
import random
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.applications.resnet_v2 import ResNet50V2
from keras.applications.resnet_v2 import preprocess_input, decode_predictions
import numpy as np
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from keras import optimizers

# path & reading the labels
data_path = os.path.abspath("C:/Users/ophir/Desktop/FlowerData/")
matlabfile = "C:/Users/ophir/Desktop/FlowerData/FlowerDataLabels"
test_images_indices = list(range(301, 473))
batch_size = 16
epochs = 11

# Basic functions:
def data_func(test_images_indices):
    '''
       The func reads the data and split it to test and train according to test_images_indices
            :param test_images_indices: range or array of test picture's numbers
            :return: 2 dictioneries, one for train and one for test. Each set is a 2-dimensional dictionary with
    images and labels
    '''
    print("Importing data")
    Labels = s.io.loadmat(matlabfile, mdict=None, appendmat=True)
    Labels = np.transpose(Labels['Labels']).tolist()  # list of labels
    train_set = {'images': [], 'labels': []}
    test_set = {'images': [], 'labels': []}
    for i in range(len(Labels)):
        img = cv2.imread(data_path + "/" + str(i + 1) + '.jpeg')
        # img = cv2.resize(img, (224, 224))
        if (i+1) in test_images_indices:
            test_set['images'].append(img)
            test_set['labels'].append(Labels[i][0])
        else:
           train_set['images'].append(img)
           train_set['labels'].append(Labels[i][0])
    return train_set, test_set

def resize_func(data_set):
    '''
       The func resizes the images of the data_set it gets to 224 / 224 as required
            :param data_set: dictionary of data (like data_func returns)
            :return: the same dictionary of data but the images are resized
    '''
    for i in range(len(data_set['images'])):
        data_set['images'][i] = cv2.resize((data_set['images'][i]), (224, 224))
    return data_set

def preprocess_for_ResNet50V2(data_set_images):
    '''
       The func prepares the data for resnet
            :param data_set_images: array of images
            :return: array of images ready for resnet
    '''
    images = []
    for img in data_set_images:
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        images.append(x)
    images = np.array(images)
    images = np.rollaxis(images, 1, 0)
    images = images[0]
    return images

def create_binary_model(num):
    '''
       The func creates the neural network
            :param num: number of layer to train. (first it was 1, and in the improve level it has changed
            :return: network with last layer in the shape we wanted ( 1 neuron)
    '''
    print("building network")
    model = ResNet50V2(include_top=False, weights='imagenet', input_tensor=None,
                       input_shape=None, pooling='avg')
    # model.trainable = False
    for layer in model.layers[:-(num)]: ## Loop over retrainable layers
        layer.trainable = False
    add = Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.01))
    new_model = keras.Sequential([model, add])
    # new_model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False),
    #                   metrics=['accuracy'])
    new_model.compile(loss='binary_crossentropy', optimizer=optimizers.Adadelta(learning_rate=1.0, rho=0.95),
                      metrics=['accuracy'])
    new_model.summary()
    return new_model

def model_train(train_set, batch_size, epochs, num):
    '''
        The func splits the train set to train and validation, then builds a network and fits it
        to the train set. in addition, it returns the validation set.
             :param train_set: train dictionary after all  preparations
             :param batch_size: hyper-parameter value
             :param epochs: hyper-parameter value
             :param num: number of layer to train
             :return: the network model after fitting to train set and a set of validation data.
     '''
    X_train, X_val, Y_train, Y_val = train_test_split(train_set['images'], train_set['labels'], test_size=0.2)
    resnet = create_binary_model(num)
    print("training network")
    model = resnet.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, Y_val), shuffle=True)
    val_set = {'images': [], 'labels': []}
    val_set['images'] = X_val
    val_set['labels'] = Y_val
    return resnet, val_set

def model_train_with_plots(train_set, batch_size, epochs,num):
    '''
        The same func as 'model_train' but with plots of loss and accuracy vs epochs for
         validation set and train set. this func are not used in the final run (i.e. the submission run)
     '''
    X_train, X_val, Y_train, Y_val = train_test_split(train_set['images'], train_set['labels'], test_size=0.2)
    resnet = create_binary_model(num)
    History = keras.callbacks.History()
    print("training network")
    model = resnet.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, Y_val), shuffle=True,
                       callbacks=[History])

    val_loss = model.history['val_loss']
    train_loss = model.history['loss']
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    plt.title('Training & validation Loss')
    plt.xlabel('epoch')
    plt.show()

    val_acc = model.history['val_accuracy']
    train_acc = model.history['accuracy']
    plt.plot(train_acc, label='Training accuracy')
    plt.plot(val_acc, label='validation accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.title('Training & validation accuracy')
    plt.xlabel('epoch')
    plt.show()

    val_set = {'images': [], 'labels': []}
    val_set['images'] = X_val
    val_set['labels'] = Y_val
    return resnet, val_set

def model_test(test_set, model, batch_size): # evaluate the model on the test
    '''
        The func evaluates the test set results on the model from the function 'model_train', and prints the
        accuracy and the loss values.
             :param test_set: test dictionary after all preparations (sometimes we use the validation set as test,
              depends on our goal)
             :param model: the network model from the function 'model_train'
             :param batch_size: hyper-parameter value
             :return: the accuracy of the model
     '''
    print("testing network")
    (loss, accuracy) = model.evaluate(test_set['images'], test_set['labels'], batch_size=batch_size)
    print("The loss is: ","%.4f" % loss , "The accuracy is: ","%.4f" % accuracy)
    return accuracy

# functions for improvements:
def crop_img(img, scale):
    '''
       The func crops the picture sent to it by the amount requested, and returns it cropped
            :param img: the original image to crop
            :param scale: the percentage to crop from the original (to sent as % * 100) (better > 80 or 90)
            :return: img_cropped: the new cropped picture
    '''
    center_x, center_y = img.shape[1] / 2, img.shape[0] / 2
    width_scaled, height_scaled = img.shape[1] * scale, img.shape[0] * scale
    left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2
    top_y, bottom_y = center_y - height_scaled / 2, center_y + height_scaled / 2
    img_cropped = img[int(top_y):int(bottom_y), int(left_x):int(right_x)]
    return img_cropped

def data_aug(train_set):
    '''
           The func makes two data augmentaions randomly, i.e. for every pic in the train set, the func flip it or crop it.
           the func add the new pictures to the train set. so now the train set length is multiply by 2.
           :param train_set: the train set pictures to make over the normal pictures
           :return: train_set: the new, doubled, augmented train_set
    '''
    len1 = len(train_set['images'])
    for i in range(0,len1):
        img = train_set['images'][i]
        r = random.randint(0, 2)
        if (r == 1):
            img = img[:, ::-1]
            # flipping img
        if (r == 2):
            img = crop_img(img, 0.85)
        img = cv2.resize(img, (224, 224))
        train_set['images'].append(img)
        train_set['labels'].append(train_set['labels'][i])
    return train_set

# Report functions:
def errorType(predictions, test_set):
    '''
         The func calculates and shows the five highest errors that the model predicted wrong. The prediction comes from
         a sigmoid function, where the predictions with score > 0.5 are predicted as flowers, and none otherwise
         0 = label 'none', 1 = label flower
         :param predictions: the scores of the model predictions for the test set
         :param test_set: the test set pictures and labels
    '''
    error_score1 = []
    error_score2 = []
    # collect the errors:
    for i in range(len(test_set['labels'])):
        if predictions[i][0] < 0.5 and test_set['labels'][i] == 1:  # type 1 error
            error_score1.append((predictions[i][0], i + 301))
        if predictions[i][0] > 0.5 and test_set['labels'][i] == 0:  # type 2 error
            error_score2.append(( predictions[i][0], i + 301))
    # sort them:
    if len(error_score1) != 0:
        error_score1 = sorted(error_score1, reverse=True)
    if len(error_score2) != 0:
        error_score2 = sorted(error_score2, reverse=False)
    # print the 5 max of each type:
    for i in range(5):
        if i< len(error_score1):
            predi, index = error_score1[i]
            print("Error type 1, Index :", index, ", Score: ", predi)
    for i in range(5):
        if i < len(error_score2):
            predi, index = error_score2[i]
            print("Error type 2, Index :", index, ", Score: ", (1 - predi))
    return

def recallPrecision(predictions, test_labels):
    '''
         The func calculates and shows the recall precision plot. Naturally, the score of the prediction will
         decrease when increasing the recalling. Again, 0 = label 'none', 1 = label flower
         :param predictions: the scores of the model predictions for the test set
         :param test_labels: the test set pictures and labels
    '''
    test_lab = []
    preds = []
    for i in range(len(test_labels['labels'])):
        test_lab.append(test_labels['labels'][i])
        preds.append(predictions[i][0])

    precision, recall, z = precision_recall_curve(test_lab, preds)
    fig = plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.xlim(0.1, 1)
    plt.ylabel('Precision')
    plt.title('Precision score VS Recall')
    plt.savefig('Recall precision curve')
    plt.show(fig)
    return

# mains functions:
def no_improve():
    '''
         The func runs the hole program, with the basic resnet50v2 network
    '''
    train_set, test_set = data_func(test_images_indices)
    test_set = resize_func(test_set)
    train_set = resize_func(train_set)
    x = preprocess_for_ResNet50V2(train_set['images'])
    y = preprocess_for_ResNet50V2(test_set['images'])
    train_set['images'] = x
    test_set['images'] = y
    model, val_set = model_train(train_set,batch_size, epochs,1)
    accuracy = model_test(test_set, model, batch_size)
    preds = model.predict(test_set['images'])
    # errorType(preds, test_set)
    recallPrecision(preds, test_set)

def aug_model():
    '''
    The func runs the hole program, with building a model on the regular and augmentation train data set.
    '''
    train_set, test_set = data_func(test_images_indices)
    n_test_set = resize_func(test_set)
    train_set = resize_func(train_set)
    n_train_set = data_aug(train_set)
    x = preprocess_for_ResNet50V2(n_train_set['images'])
    y = preprocess_for_ResNet50V2(n_test_set['images'])
    n_train_set['images'] = x
    n_test_set['images'] = y
    model, val_set = model_train(n_train_set,batch_size, epochs,1)
    accuracy = model_test(n_test_set, model, batch_size)
    preds = model.predict(n_test_set['images'])
    # errorType(preds, n_test_set)
    recallPrecision(preds, n_test_set)
    return

def train_more_then_1_layer():
    '''
    The func runs the hole program, every run with different number of trainable layer.
     finally it produced a plot that show the validation accuracy vs the number of trainable layers.
     it helps us to tuning this parameter.
    '''
    numbers = [1,2,3,4,5,6,7,8,9,10]
    accuracies = []
    for num in numbers:
        train_set, test_set = data_func(test_images_indices)
        test_set = resize_func(test_set)
        train_set = resize_func(train_set)
        x = preprocess_for_ResNet50V2(train_set['images'])
        y = preprocess_for_ResNet50V2(test_set['images'])
        train_set['images'] = x
        test_set['images'] = y
        model, val_set = model_train(train_set, batch_size, epochs, num)
        accuracy = model_test(val_set, model, batch_size)
        accuracies.append(accuracy)

    plt.plot(accuracies, label='validation accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.title('Validation accuracy')
    plt.xlabel('number of trainable layers')
    plt.show()
    return

def train_8_layer():
    '''
    The func runs the hole program, with a model with 8 trainable layers.
    '''
    train_set, test_set = data_func(test_images_indices)
    test_set = resize_func(test_set)
    train_set = resize_func(train_set)
    x = preprocess_for_ResNet50V2(train_set['images'])
    y = preprocess_for_ResNet50V2(test_set['images'])
    train_set['images'] = x
    test_set['images'] = y
    model, val_set = model_train(train_set, batch_size, epochs, 8)
    accuracy = model_test(test_set, model, batch_size)
    return

def train_8_layer_plus_aug():
    '''
     The func runs the hole program, with a model with 8 trainable layers and with augmentation data train
     '''
    train_set, test_set = data_func(test_images_indices)
    n_test_set = resize_func(test_set)
    train_set = resize_func(train_set)
    n_train_set = data_aug(train_set)
    x = preprocess_for_ResNet50V2(n_train_set['images'])
    y = preprocess_for_ResNet50V2(n_test_set['images'])
    n_train_set['images'] = x
    n_test_set['images'] = y
    model, val_set = model_train(n_train_set,batch_size, epochs,8)
    accuracy = model_test(n_test_set, model, batch_size)
    preds = model.predict(n_test_set['images'])
    # errorType(preds, n_test_set)
    recallPrecision(preds, n_test_set)
    return

# -------------------------- main: -------------------------- #

np.random.seed(0)
# no_improve()
# aug_model()
# train_more_then_1_layer()
# train_8_layer()
train_8_layer_plus_aug()