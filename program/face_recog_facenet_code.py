from keras.models import load_model
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from random import choice
import shutil

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
DIR_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)))[:-21]
DIR_RES = os.path.join(DIR_BASE, 'database/exports')
DIR_FACES = os.path.join(DIR_RES, 'cropped_faces')
DIR_OCCLUDED_FACES = os.path.join(DIR_RES, 'occluded_faces')
DIR_DATABASE = os.path.join(DIR_RES, 'database')
DIR_NET = os.path.join(DIR_BASE, 'database/neuralnet')


class DatasetProcessing:
    def __init__(self, data_dir, save_path, split_location = (6, 10)):
        self.data_dir = data_dir
        self.save_path = save_path
        self.split_location = split_location

    def process_imgdata(self):
        train_path, train_label, test_path, test_label = self.get_image_paths()
        trainX, trainY = self.set_datasets(train_path, train_label)
        testX, testY = self.set_datasets(test_path, test_label)
        np.savez_compressed(self.save_path, trainX, trainY, testX, testY)
        print(f'DATA PROCESSED : {trainX.shape}, {trainY.shape}, {testX.shape}, {testY.shape}')

    def get_image_paths(self):
        temp = [[], [], [], []]
        x, y = self.split_location
        for root, dir, files in os.walk(self.data_dir):
            if len(files) > 0:
                for img in files:
                    c = int(img[:-4])
                    img = os.path.join(root, img)
                    if x <= c <= y:
                        temp[2].append(img)
                        temp[3].append(root[-3:])
                    else:
                        temp[0].append(img)
                        temp[1].append(root[-3:])

        return temp[0], temp[1], temp[2], temp[3]


    def set_datasets(self, img_paths, img_labels):
        x, y = list(), list()
        image = [self.preprocess_image(img) for img in img_paths]
        x.extend(image)
        y.extend(img_labels)
        return np.asarray(x), np.asarray(y)

    def preprocess_image(self, filepath, size=(160, 160)):
        image = Image.open(filepath).convert('RGB')
        image = Image.fromarray(np.array(image)).resize(size)
        return np.asarray(image)

class FaceEmbedding:
    def __init__(self, model, data, save_path):
        self.model = model
        self.data = np.load(data)
        self.save_path = save_path

    def process_face_embedding(self):
        trainX = self.data['arr_0']
        trainY = self.data['arr_1']
        testX = self.data['arr_2']
        testY = self.data['arr_3']
        print('Loaded: ', trainX.shape, trainY.shape, testX.shape, testY.shape)

        emb_trainX = self.convert_to_embedding(trainX)
        emb_testX = self.convert_to_embedding(testX)

        # save arrays to on file in compressed format
        np.savez_compressed(self.save_path, emb_trainX, trainY, emb_testX, testY)
        print('EMBEDDING COMPLETE')

    def get_embedding(self, pixel):
        pixels = pixel.astype('float32')
        # standardize pixel values across channels
        mean = pixels.mean()
        std = pixels.std()
        pixels = (pixels - mean) / std
        # transfrom into one sample
        samples = np.expand_dims(pixels, axis=0)
        # make prediction to get embedding
        yhat = self.model.predict(samples)
        return yhat[0]

    def convert_to_embedding(self, image):
        embeddings = list(self.get_embedding(px) for px in image)
        return np.asarray(embeddings)

class FaceClassifier:
    def __init__(self, data, model):
        self.data = np.load(data)
        self.model = model
        # accuracy of norm, abstract, and mouth

    def set_data(self):
        trainX = self.data['arr_0']
        trainY = self.data['arr_1']
        testX = self.data['arr_2']
        testY = self.data['arr_3']

        # normalize input vectors
        input_encoder = Normalizer(norm='l2')
        trainX = input_encoder.transform(trainX)
        testX = input_encoder.transform(testX)

        # label encode targets
        self.output_encoder = LabelEncoder()
        self.output_encoder.fit(trainY)
        trainY = self.output_encoder.transform(trainY)
        testY = self.output_encoder.transform(testY)

        # fit model
        self.model = SVC(kernel='linear', probability=True)
        self.model.fit(trainX, trainY)

        return trainX, trainY, testX, testY

    def get_accuracy(self):
        trainX, trainY, testX, testY = self.set_data()
        yhat_train = self.model.predict(trainX)
        yhat_test = self.model.predict(testX)
        acc_train = accuracy_score(trainY, yhat_train)
        acc_test = accuracy_score(testY, yhat_test)

        print(f'DATASET: train : {trainX.shape[0]}'
              f'\n       test  : {testX.shape[0]}')

        print(f'ACCURACY: train :{acc_train*100} '
              f'\n          test  : {acc_test*100} ')

    def plot_predicted(self):
        data = np.load('face_recog_masked.npz')
        # testX_faces = data['arr_2']

        trainX, trainY, testX, testY = self.set_data()

        label_match = 0
        for i in range(testX.shape[0]):
            image_embed = testX[i]
            face_label = self.output_encoder.inverse_transform([testY[i]])

            # predictions
            samples = np.expand_dims(image_embed, axis=0)
            yhat_label = self.model.predict(samples)
            yhat_prob = self.model.predict_proba(samples)

            # get label
            label_probability = yhat_prob[0, yhat_label[0]] * 100
            predict_label = self.output_encoder.inverse_transform(yhat_label)
            # print(f'Predicted : {predict_label}, {label_probability}')
            # print(f'Expected: {face_label}')

            if face_label == predict_label:
                label_match += 1
        print(f'Matched total: {label_match} : {testX.shape[0]} = {(label_match / testX.shape[0]) * 100}')


    def prediction_accuracy_matching(self):
        data = np.load('face_recog_masked.npz')
        testX_faces = data['arr_2']

        trainX, trainY, testX, testY = self.set_data()

        # test model on a random example from the test dataset
        selection = choice([i for i in range(testX.shape[0])])
        image = testX_faces[selection]
        image_embed = testX[selection]
        image_label = testY[selection]
        face_label = self.output_encoder.inverse_transform([image_label])

        # prediction for the face
        samples = np.expand_dims(image_embed, axis=0)
        yhat_label = self.model.predict(samples)
        yhat_prob = self.model.predict_proba(samples)

        # get label
        label_index = yhat_label[0]
        label_probability = yhat_prob[0, label_index] * 100
        predict_label = self.output_encoder.inverse_transform(yhat_label)
        print('Predicted: %s (%.3f)' % (predict_label, label_probability))
        print('Expected: %s' % face_label)
        if face_label == predict_label: print('True')
        # plot
        plt.imshow(image)
        plt.title(f'{predict_label[0], label_probability}')
        plt.show()




# get face and occluded face image paths for training
face_img_paths = []
face_img_paths_2 = []
# occluded_face_img_paths = []
# face_dir = DIR_OCCLUDED_FACES
face_dir = DIR_FACES
# face_dir = os.path.join(DIR_RES, 'eye_brow_only_faces')
# face_dir2 = DIR_OCCLUDED_FACES
face_dir2 = os.path.join(DIR_RES, 'eye_brow_only_faces')
# face_dir2 = DIR_FACES
for root, dirs, files in os.walk(face_dir):
    if len(files) > 0:
        for img in files:
            face_img_paths.append(os.path.join(root, img))
            face_img_paths_2.append(os.path.join(face_dir2, root[-3:], img))


# build image dataset paths of testing
image_path_list = [[],[],[]]
mask_dir_names = ['mask_norm', 'mask_mouth', 'mask_abstract']
for i in range(3):
    c = 0
    # test_path = os.path.join(DIR_RES, mask_dir_names[i])
    test_path = DIR_FACES
    for root, dir, files in os.walk(test_path):
        if len(files) > 0:
            for j in range(len(files)):
                # print(root)
                # if j < 5 or j > 9: image_path_list[i].append(face_img_paths[c])
                if j < 5: image_path_list[i].append(face_img_paths[c])
                elif j > 9: image_path_list[i].append(face_img_paths_2[c])
                else: image_path_list[i].append(os.path.join(root, files[j]))
                c += 1
# duplicate images
d_dir_ls = []
database_name = 'masked_database'
for dmp in range(len(mask_dir_names)):
    c = 0
    for mp in range(len(image_path_list[dmp])):
        if mp % 15 == 0 : c += 1
        dirname = os.path.join(DIR_RES, database_name,
                               mask_dir_names[dmp], 's' + str(c).zfill(2))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        shutil.copy2(image_path_list[dmp][mp], dirname)
    d_dir_ls.append(os.path.join(DIR_RES, database_name, mask_dir_names[dmp]))



'''

#### FOR BUILDING / DUPLICATING THE IMAGE DATASET

# image_path_list = [[],[],[]]
# face_img_paths = []
# for root, dirs, files in os.walk(DIR_FACES):
#     if len(files) > 0:
#         for img in files:
#             face_img_paths.append(os.path.join(root, img))
# # print(face_img_paths[0])
# 
# mask_dir_names = ['mask_norm', 'mask_mouth', 'mask_abstract']
# for i in range(3):
#     c = 0
#     for root, dir, files in os.walk(os.path.join(DIR_RES, mask_dir_names[i])):
#         if len(files) > 0:
#             for j in range(len(files)):
#                 # print(root)
#                 if j < 5 or j > 9: image_path_list[i].append(face_img_paths[c])
#                 else: image_path_list[i].append(os.path.join(root, files[j]))
#                 c += 1
# for img in image_path_list[0]:
#     print(img)
# d_dir_ls = []
# for dmp in range(len(mask_dir_names)):
#     c = 0
#     for mp in range(len(image_path_list[dmp])):
#         if mp % 15 == 0 : c += 1
#         dirname = os.path.join(DIR_RES, database_name,
#                                mask_dir_names[dmp], 's' + str(c).zfill(2))
#         if not os.path.exists(dirname):
#             os.makedirs(dirname)
#         shutil.copy2(image_path_list[dmp][mp], dirname)
#     d_dir_ls.append(os.path.join(DIR_RES, database_name, mask_dir_names[dmp]))
'''



def main():


    datafile_dir = os.path.join(DIR_NET, 'facenet')
    model = load_model(os.path.join(datafile_dir, 'facenet_keras.h5'))

    data_name = ('mask_norm', 'mask_abstract', 'mask_mouth')
    image_database_dir = (
        os.path.join(DIR_RES, 'masked_database', data_name[0]),
        os.path.join(DIR_RES, 'masked_database', data_name[1]),
        os.path.join(DIR_RES, 'masked_database', data_name[2]) )
    dataset_file = ('facenet_masked_normal_database.npz',
                  'facenet_masked_abstract_database.npz',
                  'facenet_masked_mouth_database.npz')
    embed_name = ('facenet_masked_normal_embeddings.npz',
                  'facenet_masked_abstract_embeddings.npz',
                  'facenet_masked_mouth_embeddings.npz')
    for i in range(len(dataset_file)):
        print('\n========================')
        print(f'PROCESSING DATA {data_name[i]}')
        DatasetProcessing(
            data_dir=image_database_dir[i],
            save_path=os.path.join(datafile_dir, dataset_file[i]),
        ).process_imgdata()

        print(f'EMBEDDING DATA {data_name[i]}')
        FaceEmbedding(
            model=model, data=os.path.join(datafile_dir, dataset_file[i]),
            save_path=os.path.join(datafile_dir, embed_name[i])
        ).process_face_embedding()

        print(f'CLASSIFIYING DATA {data_name[i]}')
        FC = FaceClassifier(
            data=os.path.join(datafile_dir, embed_name[i]),
            model=model)

        FC.get_accuracy()
        FC.plot_predicted()
        print('========================')


if __name__ == '__main__':
    main()
















