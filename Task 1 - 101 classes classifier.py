import os
import cv2
import numpy
from matplotlib import pyplot as plt
from sklearn import svm
import sklearn
from skimage.feature import hog
from sklearn.metrics import plot_confusion_matrix

# class_indices is at the bottom of the code, in the main, as required
data_path = os.path.abspath("C:/Users/ophir/Desktop/101_ObjectCategories")

# ----- help functions ----- :

def get_default_parameters(data_path, class_indices, s, pixels_per_cell, cells_per_block, orientation, kernel, c, gamma, degree):
  '''
  The func create a dictionary with all the fixed hyper-parameters
   :param data_path: the path for the folders
   :param class_indices: indices of the folders we want to use
   :param s: size of image
   :param pixels_per_cell: hog parameter
   :param cells_per_block: hog parameter
   :param orientation: hog parameter
   :param kernel: the kernel the SVM will use
   :param c: SVM "regulator" parameter
   :param gamma: rbf kernel parameter
   :param degree: degree of the polynom in poly kernel
   :return: a dictionary with all the fixed hyper-parameters
  '''
  params = {}
  params['data_path'] = data_path
  params['class_indices'] = class_indices
  params['size'] = s
  params['pixels_per_cell'] = pixels_per_cell
  params['cells_per_block'] = cells_per_block
  params['orientation'] = orientation
  params['kernel'] = kernel
  params['c'] = c
  params['gamma'] = gamma
  params['degree'] = degree
  return params

def data_func(class_indices, s):
  '''
  The func read the data and prepare it for the pipe
   :param class_indices: indices of the folders we want to use
   :param s: size of image
   :return: 3 sets of data in partition to train, validation and test. Each set is a 2-dimensional dictionary with
    images and labels
  '''
  folders = os.listdir(data_path)
  folders.sort(key=lambda v: v.upper())
  class_indices.sort()
  train_set = {'images':[], 'labels':[]}
  validation_set = {'images':[], 'labels':[]}
  test_set = {'images':[], 'labels':[]}
  labels_names = []
  for i in class_indices:
     labels_names.append(folders[i-1])
     images = os.listdir(data_path+"/"+folders[i-1])
     images.sort(key=lambda v: v.upper())
     p=0
     while p<40 and p<len(images):
        new_image = cv2.imread(data_path+"/" +folders[i-1]+"/" +images[p],0)
        new_image = cv2.resize(new_image, (s,s), interpolation=cv2.INTER_CUBIC)
        if(p<16):
          train_set['images'].append(new_image)
          train_set['labels'].append(folders[i-1])
        elif(p<20):
          validation_set['images'].append(new_image)
          validation_set['labels'].append(folders[i-1])
        else:
          test_set['images'].append(new_image)
          test_set['labels'].append(folders[i-1])
        p=p+1
  return train_set, validation_set, test_set

def hog_func(data, orient, ppc, cpb):
 '''
    The func get the images and convert them to hog representation.
     :param data: the data for hoging
     :param orient: the value of this hyper-parameter
     :param ppc: the value of this hyper-parameter
     :param cpb: the value of this hyper-parameter
     :return: 3-dimensional dictionary with the images, the hog features and the labels.
 '''
 HOGS = {'images': [], 'hogs': [], 'labels': []}
 for i in range(len(data['images'])):
  image = data['images'][i]
  HOG = hog(image, orientations=orient, pixels_per_cell=(ppc, ppc), cells_per_block=(cpb, cpb),
            block_norm='L1', visualize=False, transform_sqrt=False,
            feature_vector=True, multichannel=None)
  HOGS['images'].append(image)
  HOGS['hogs'].append(HOG)
  HOGS['labels'].append(data['labels'][i])
 return HOGS

def any_svm(curr_tr_set, kernel, c, gamma, degree):
    '''
       The func get the data and runs the SVM algorithm according to the input parameters.
        :param curr_tr_set: the train set for the svm
        :param kernel: the chosen kernel (linear/ rbf / poly)
        :param c: the value of this svm hyper-parameter
        :param gamma: the value of this hyper-parameter ( will be used if kernel=rbf)
        :param degree: the value of this hyper-parameter( will be used if kernel=poly)
        :return: the svm model according to this train data.
    '''
    SVC = svm.SVC(kernel=kernel, C=c, gamma=gamma,  degree=degree, probability=True)
    clf = SVC.fit(curr_tr_set['hogs'], curr_tr_set['labels'])
    return clf

def m_class_SVM_train(tr_set, kernel, c, gamma, degree):
    '''
       The func do 'one VS all' svm for the classes in the train set.
        :param tr_set: the hole train set for the svm ( will be split in the function)
        :param kernel: the chosen kernel (linear/ rbf / poly)
        :param c: the value of this svm hyper-parameter
        :param gamma: the value of this hyper-parameter ( will be used if kernel=rbf)
        :param degree: the value of this hyper-parameter( will be used if kernel=poly)
        :return: a dictionary with the m svm models and the labels (i.e. the label of the "one" from the
         one VS all that fit to the svm model).
    '''
    SVM_models = []
    labels = []
    hole_labels = tr_set['labels']
    uni_labels = list(set(hole_labels))
    uni_labels.sort(key=lambda v: v.upper())
    for uni_label in uni_labels:
        new_labels = []
        for label in hole_labels:
            if label == uni_label:
                new_labels.append(1)
            else:
                new_labels.append(-1)
        curr_train_set = {'images': tr_set['images'], 'hogs': tr_set['hogs'], 'labels': new_labels}
        SVM_models.append(any_svm(curr_train_set, kernel, c, gamma, degree))
        labels.append(uni_label)
    m_classes_SVM = {'features': SVM_models, 'labels': labels}
    return m_classes_SVM

def m_class_SVM_predict(m_classes_SVM, test_set):
    '''
       The func do prediction to the data using the svm models.
        :param m_classes_SVM: a dictionary with the m svm models and the labels (the return from the previous function))
        :param test_set: the test set to classify
        :return: score_matrix: a matrix that contain for every test image - the probability to belong to each class
        :return: predict: vector of prediction -for every test image the chosen label (the class with the max
                  probability to belong)
    '''
    hole_labels = test_set['labels']
    uni_labels = list(set(hole_labels))
    uni_labels.sort(key=lambda v: v.upper())
    num_of_images = len(test_set['images'])
    predict = []
    score_matrix = numpy.zeros((num_of_images, len(uni_labels)))
    for j in range(len(uni_labels)):
        probability = (m_classes_SVM['features'][j]).predict_proba(test_set['hogs'])
        for i in range(len(probability)):
            score_matrix[i, j] = probability[i, 1]
    for num in range(num_of_images):
        results_per_image = score_matrix[num, :]
        max_index = numpy.argmax(results_per_image)
        predict.append(uni_labels[max_index])
    return score_matrix, predict


def svm_results_func(predict, test_labels):
 '''
 The func calculate the results of the prediction
    :param predict: vector of the predicted classes (labels) from the svm algo
    :param test_labels: the true labels of the test set
    :return: error_rate: the sum of the classification errors divided by the number of images
    :return: confusion_matrix: for every class (rows), the number of classifications for each class(columns)
     (10*10 matrix)
 '''
 confusion_matrix = sklearn.metrics.confusion_matrix(test_labels, predict)
 error = 0
 for i in range(len(predict)):
  if (predict[i] != test_labels[i]):
    error = error + 1
 error_rate = error / len(test_labels)
 return error_rate, confusion_matrix

def find_2_max_errors(score_matrix, test_set):
    '''
    The func create a list of indexes of the 2 largest error images in any class using margin calculation.
       :param score_matrix: a matrix that contain for every test image - the probability to belong to each class
       :param test_set: the test dictionary (images, hogs, labels)
       :return: list_of_max_errors_images: list of the indexes of the 2 images with the biggest classification mistake
        (lowest margin) in each class (if exists)
    '''
    hole_labels = test_set['labels']
    uni_labels = list(set(hole_labels))
    uni_labels.sort(key=lambda v: v.upper())
    # calc margins:
    margins = {'labels':[], 'margins':[]}
    margins['labels']=test_set['labels']
    for i in range(len(test_set['labels'])):
        num = 0
        for n in range(len(uni_labels)):
            if test_set['labels'][i] == uni_labels[n]:
                num = n
        margins['margins'].append(score_matrix[i, num] - numpy.amax(score_matrix[i, :]))
    # find indexes:
    list_of_max_errors_images =[]
    u=0
    j=0
    while u <(len(margins['margins'])):
        count = margins['labels'].count(uni_labels[j])
        max1 = min(margins['margins'][u:(u + count)])
        max_ind = margins['margins'][u:(u + count)].index(min(margins['margins'][u:(u + count)]))
        if (max1 < 0):
            list_of_max_errors_images.append(max_ind + u)
            margins['margins'][max_ind + u] = 30
            max2 = min(margins['margins'][u:(u + count)])
            max2_ind = margins['margins'][u:(u + count)].index(min(margins['margins'][u:(u + count)]))
            if (max2 < 0):
                list_of_max_errors_images.append(max2_ind + u)
            else:
                list_of_max_errors_images.append(None)
        else:
            list_of_max_errors_images.append(None)
            list_of_max_errors_images.append(None)
        u = u + count
        j = j + 1
    return list_of_max_errors_images


def show_error_images(list_of_max_errors_images, set_test):
    '''
     The func show the largest error images in one page.
        :param list_of_max_errors_images: the list of the indexes of the error images
        :param set_test: the test data which the images are there
     '''
    fig = plt.figure(figsize=(8, 8))
    columns = 4
    rows = 5
    for i in range(len(list_of_max_errors_images)):
        if (list_of_max_errors_images[i] != None):
            image = set_test['images'][list_of_max_errors_images[i]]
            plt.axis('off')
            fig.add_subplot(rows, columns, i+1)
            plt.imshow(image, cmap = 'gray', interpolation = 'bicubic')
    plt.axis('off')
    plt.show()

def tuning_main_func():
 '''
    The main func that we use for tuning the hyper parameters.
 '''
 params = get_default_parameters(data_path, class_indices, s=100, pixels_per_cell=8, cells_per_block=2,
                               orientation=9, kernel='linear', c=1, gamma='scale', degree=2)
 numpy.random.seed(0)

 # ----- linear svm ----- :
 # find s:
 s_errors = []
 s_range = numpy.arange(90, 301, 30)
 print(s_range)
 for s in s_range:
     train, validation, test = data_func(fold1, s)
     hog_train = hog_func(train, params['orientation'], params['pixels_per_cell'], params['cells_per_block'])
     linear_svm_models = any_svm(hog_train,  params['kernel'], params['c'], params['gamma'], params['degree'])
     hog_test = hog_func(validation, params['orientation'], params['pixels_per_cell'], params['cells_per_block'])
     predict = linear_svm_models.predict(hog_test['hogs'])
     error_rate, con_matrix = svm_results_func(predict, hog_test['labels'])
     s_errors.append(error_rate)
 print(s_errors)
 plt.plot(s_range, s_errors, color='lightblue', linewidth=2)
 plt.title('Liner SVM - Size vs Error Rate')
 plt.xlabel('s')
 plt.ylabel('Error rate')
 plt.show()

 # find ppc, cpb:
 ppc_cpb_errors = []
 ppc_range = numpy.arange(10, 31, 5)
 cpb_range = numpy.arange(2, 6, 1)
 print(ppc_range)
 print(cpb_range)
 train, validation, test = data_func(fold1, 240)
 for p in ppc_range:
   for c in cpb_range:
       hog_train = hog_func(train, params['orientation'], p, c)
       linear_svm_models = any_svm(hog_train, params['kernel'], params['c'], params['gamma'], params['degree'])
       hog_test = hog_func(validation, params['orientation'], p, c)
       predict = linear_svm_models.predict(hog_test['hogs'])
       error_rate, con_matrix = svm_results_func(predict, hog_test['labels'])
       ppc_cpb_errors.append(error_rate)
 print(ppc_cpb_errors)

# find orient:
 orient_errors = []
 orient_range = numpy.arange(5, 51, 5)
 train, validation, test = data_func(fold1, 240)
 for o in orient_range:
     hog_train = hog_func(train, o, 30, 3)
     linear_svm_models = any_svm(hog_train, params['kernel'], params['c'], params['gamma'], params['degree'])
     hog_test = hog_func(validation, o, 30, 3)
     predict = linear_svm_models.predict(hog_test['hogs'])
     error_rate, con_matrix = svm_results_func(predict, hog_test['labels'])
     orient_errors.append(error_rate)
 print(orient_errors)
 plt.plot(orient_range, orient_errors, color='red', linewidth=2)
 plt.title('Liner SVM - Orientation vs Error Rate')
 plt.xlabel('Orientation')
 plt.ylabel('Error rate')
 plt.show()

# find c
 c_errors = []
 c_range = numpy.logspace(-9, 3, 13)
 train, validation, test = data_func(fold1, 240)
 for c in c_range:
     hog_train = hog_func(train, 20, 30, 3)
     linear_svm_models = any_svm(hog_train, params['kernel'], c, params['gamma'], params['degree'])
     hog_test = hog_func(validation, 20, 30, 3)
     predict = linear_svm_models.predict(hog_test['hogs'])
     error_rate, con_matrix = svm_results_func(predict, hog_test['labels'])
     c_errors.append(error_rate)

# ----- non linear svm - (rbf kernel as default) ----- :
# find s:
 s_errors = []
 s_range = numpy.arange(90, 301, 30)
 for s in s_range:
     train, validation, test = data_func(fold1, s)
     hog_train = hog_func(train, params['orientation'], params['pixels_per_cell'], params['cells_per_block'])
     non_linear_svm_models = m_class_SVM_train(hog_train, 'rbf', params['c'], params['gamma'], params['degree'])
     hog_test = hog_func(validation, params['orientation'], params['pixels_per_cell'], params['cells_per_block'])
     score_matrix, predictions = m_class_SVM_predict(non_linear_svm_models, hog_test)
     error_rate, con_matrix = svm_results_func(predictions, hog_test['labels'])
     s_errors.append(error_rate)
 print(s_errors)
 plt.plot(s_range, s_errors, color='lightblue', linewidth=2)
 plt.title('Non-liner SVM (rbf) - Size vs Error Rate')
 plt.xlabel('S')
 plt.ylabel('Error rate')
 plt.show()

# find ppc, cpb:
 ppc_cpb_errors = []
 ppc_range = numpy.arange(10, 31, 5)
 cpb_range = numpy.arange(2, 6, 1)
 train, validation, test = data_func(fold1, 210)
 for p in ppc_range:
   for c in cpb_range:
       hog_train = hog_func(train, params['orientation'], p, c)
       non_linear_svm_models = m_class_SVM_train(hog_train, 'rbf', params['c'], params['gamma'], params['degree'])
       hog_test = hog_func(validation, params['orientation'], p, c)
       score_matrix, predictions = m_class_SVM_predict(non_linear_svm_models, hog_test)
       predict = non_linear_svm_models.predict(hog_test['hogs'])
       error_rate, con_matrix = svm_results_func(predictions, hog_test['labels'])
       ppc_cpb_errors.append(error_rate)
 print(ppc_cpb_errors)

# find orient:
 orient_errors = []
 orient_range = numpy.arange(5, 51, 5)
 print(orient_range)
 for o in orient_range:
     train, validation, test = data_func(fold1, 210)
     hog_train = hog_func(train, o, 30, 3)  # change p and c
     non_linear_svm_models = m_class_SVM_train(hog_train, 'rbf', params['c'], params['gamma'], params['degree'])
     hog_test = hog_func(validation, o, 30, 3)
     score_matrix, predictions = m_class_SVM_predict(non_linear_svm_models, hog_test)
     error_rate, con_matrix = svm_results_func(predictions, hog_test['labels'])
     orient_errors.append(error_rate)
 print(orient_errors)
 plt.plot(orient_range, orient_errors, color='red', linewidth=2)
 plt.title('Non-liner SVM (rbf) - Orientation vs Error Rate')
 plt.xlabel('Orientation')
 plt.ylabel('Error rate')
 plt.show()

# find c & gamma - 'rbf'
 c_g_errors = []
 c_range = numpy.logspace(-9, 3, 13)
 gamma_range = numpy.logspace(-9, 3, 13)
 train, validation, test = data_func(fold1, 210)
 for c in c_range:
     for g in gamma_range:
         hog_train = hog_func(train, 10, 30, 3)
         non_linear_svm_models = m_class_SVM_train(hog_train, 'rbf', c, g, params['degree'])
         hog_test = hog_func(validation, 10, 30, 3)
         score_matrix, predictions = m_class_SVM_predict(non_linear_svm_models,hog_test)
         error_rate, con_matrix = svm_results_func(predictions, hog_test['labels'])
         c_g_errors.append(error_rate)

# find c & degree - 'poly'
 c_d_errors = []
 c_range = numpy.logspace(-9, 3, 13)
 degree_range = numpy.arange(2, 6, 1)
 train, validation, test = data_func(fold1, 210)
 for c in c_range:
     for d in degree_range:
         hog_train = hog_func(train, 10, 30, 3)
         non_linear_svm_models = m_class_SVM_train(hog_train, 'poly', c, params['gamma'], d)
         hog_test = hog_func(validation, 10, 30, 3)
         score_matrix, predictions = m_class_SVM_predict(non_linear_svm_models,hog_test)
         error_rate, con_matrix = svm_results_func(predictions, hog_test['labels'])
         c_d_errors.append(error_rate)


def test_main_func():
  '''
   The main func that we use for test.
  '''
  params = get_default_parameters(data_path, class_indices, 240, 30, 3, 20, kernel='linear', c=1, gamma=7, degree=2)
  numpy.random.seed(0)
  train, validation, test = data_func(class_indices, params['size'])

  total_train = {'images': [], 'labels': []}
  total_train['images'] = train['images'] + validation['images']
  total_train['labels'] = train['labels'] + validation['labels']

  hog_train = hog_func(total_train, params['orientation'], params['pixels_per_cell'], params['cells_per_block'])
  svm_models = m_class_SVM_train(hog_train, params['kernel'], params['c'], params['gamma'], params['degree'])

  hog_test = hog_func(test, params['orientation'], params['pixels_per_cell'], params['cells_per_block'])
  score_matrix, predictions = m_class_SVM_predict(svm_models, hog_test)

  error_rate, con_matrix = svm_results_func(predictions, hog_test['labels'])
  list_2_max_errors = find_2_max_errors(score_matrix, hog_test)

  print('The total error rate is: ', error_rate)
  print('The confusion matrix is: ', '\n', con_matrix)
  show_error_images(list_2_max_errors, hog_test)

# ----- Main: -----
fold1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
fold2 = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# class_indices = fold1
# tuning_main_func()
class_indices = fold2
test_main_func()


