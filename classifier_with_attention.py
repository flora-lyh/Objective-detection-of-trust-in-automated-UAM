import pandas as pd
import mne
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Convolution2D, Activation, MaxPool2D, Flatten, Dense, BatchNormalization, \
    Dropout, Reshape, Input, Layer, Permute, Multiply, Lambda, multiply, Concatenate
from tensorflow.keras.models import load_model, save_model
from sklearn.model_selection import ShuffleSplit, KFold, train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import model_from_json
from sklearn.metrics import f1_score, precision_score, recall_score
import tensorflow as tf
import os
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import keras
from tensorflow.keras import layers
from keras import regularizers
from sklearn.metrics import accuracy_score

normal_1 = pd.read_csv('OddballData0114/normal_wave.csv')
normal_2 = pd.read_csv('OddballData0114/normal_wave_1.csv')
normal_wave = np.concatenate((normal_1, normal_2), axis = 0)
print('normal wave:', normal_wave.shape)
no1 = pd.read_csv('OddballData0114/no1.csv')
no2 = pd.read_csv('OddballData0114/no2.csv')
no = np.concatenate((no1, no2), axis = 0)
print('no number:', no.shape)
low1 = pd.read_csv('OddballData0114/low1.csv')
low2 = pd.read_csv('OddballData0114/low2.csv')
low = np.concatenate((low1, low2), axis = 0)
print('low number:', low.shape)
high = pd.read_csv('OddballData0114/HIGH.csv')
print('high number:', high.shape)
df = np.concatenate((normal_1, normal_2, no1, no2, low1, low2, high), axis = 0)

label = np.array(df[:, 0])
#time_stamp = df.values[:, 1]
eeg_data_raw = np.array(df[:, 2:27].astype(float))
eeg_data = eeg_data_raw.transpose()
print(eeg_data.shape)

ch_names = ['CPz', 'Pz', 'Fz', 'FCz', 'Cz', 'C3', 'C1', 'CP3', 'P3', 'F3', 'FC3', 'C5', 'C4', 'C6', 'CP4', 'P4', 'F4', 'FC4', 'C2', 'PO6', 'O1', 'O2', 'PO5', 'PO3', 'PO4']
ch_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg']
sfreq = 1000  # Hz

info = mne.create_info(ch_names, sfreq, ch_types)
raw = mne.io.RawArray(eeg_data, info)
print('数据集的形状为：', raw.get_data().shape)
print('通道数为：', raw.info.get('nchan'))

montage = mne.channels.make_standard_montage("standard_1020")
raw.set_montage(montage)

# 创建 events & event_id
events = np.zeros((5055, 3), dtype='int')
k = sfreq * 0
for i in range(5055):
    events[i, 0] = k
    k += sfreq * 1
    a = 1000*i
    events[i, 2] = label[a]

event_id = dict(normal_wave = 0, non_workload = 1, low_workload = 2, high_workload = 3)

# 创建epochs
tmin, tmax = -0.0, 0.7  # 记录点的前0秒后0.8秒用于生成epoch数据
epochs = mne.Epochs(raw, events, event_id
                    , tmin, tmax
                    , proj=True
                    , baseline=(0, 0)
                    , preload=True
                    )

labels = epochs.events[:, -1]

#特征提取和分类
scores = []
epochs_data = epochs.get_data()                       #获取epochs的所有数据，主要用于后面的滑动窗口验证
print('epochs_data 的形状是', epochs_data.shape)

# 划分训练集，测试集
encoder = LabelEncoder()
label_encoded = encoder.fit_transform(labels)
label_onehot = np_utils.to_categorical(label_encoded)
epochs_data_expend = np.expand_dims(epochs_data, axis=3)
X_train, X_test, Y_train, Y_test = train_test_split(epochs_data_expend, label_onehot, test_size=0.2, random_state=42)
print('训练集形状：', X_train.shape)

class Metrics(tf.keras.callbacks.Callback):
    def __init__(self, valid_data):
        super(Metrics, self).__init__()
        self.validation_data = valid_data

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_predict = tf.argmax(self.model.predict(self.validation_data[0]), -1)
        val_targ = self.validation_data[1]
        if len(val_targ.shape) == 2 and val_targ.shape[1] != 1:
            val_targ = np.argmax(val_targ, -1)

        _val_f1 = f1_score(val_targ, val_predict, average='macro')
        _val_recall = recall_score(val_targ, val_predict, average='macro')
        _val_precision = precision_score(val_targ, val_predict, average='macro')

        logs['val_f1'] = _val_f1
        logs['val_recall'] = _val_recall
        logs['val_precision'] = _val_precision
        print(" — val_f1: %f — val_precision: %f — val_recall: %f" % (_val_f1, _val_precision, _val_recall))
        return
checkpoint = ModelCheckpoint('best_weights0401.h5', monitor='val_accuracy', mode='max', save_best_only=True)
early_stopping=keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0,
                              patience=50, verbose=1, mode='max',
                              baseline=None, restore_best_weights=True)

#建立网络
# 定义注意力层
# def attention_block(inputs, name='attention_block'):
    # First we need to squeeze the spatial dimension (HxW) into a single channel (C)
# x = Permute((2, 3, 1))(inputs)
# shape = x.shape.as_list()
# x = Reshape((shape[1], shape[2]*shape[3]))(inputs)
# x = Dense(shape[2], activation='softmax')(x)
# x = Reshape((shape[2], shape[1], 1))(x)
# Apply attention to the input features
# x = Multiply()([inputs, x])
#    x = Reshape((shape[1], shape[2], shape[3]))(x)
#    return x

class Attention(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels):
        super(Attention, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(out_channels//2, kernel_size=1)
        self.conv2 = tf.keras.layers.Conv2D(1, kernel_size=1)

    def call(self, inputs):
        # Unpack input tensor(s)
        query, value, key = inputs

        # Apply attention mechanism
        attn_logits = tf.matmul(query, key, transpose_b=True)
        attn = tf.nn.softmax(attn_logits)
        attn = tf.matmul(attn, value)

        # Add skip connection and return
        return attn + value

    def get_config(self):
        config = super(Attention, self).get_config()
        return config

def baseline_model():
    input_layer = Input(shape=(25, 701, 1))
    x = Convolution2D(filters=256, kernel_size=3, strides=2, padding='same')(input_layer)
    x = MaxPool2D(pool_size=(2, 2), strides=None, padding='same')(x)
    x = BatchNormalization()(x)
    x = Convolution2D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=None, padding='same')(x)
    x = BatchNormalization()(x)
    x = Convolution2D(filters=64, kernel_size=3, padding='same')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=None, padding='same')(x)
    x = BatchNormalization()(x)

    query = Convolution2D(filters=16, kernel_size=1, padding='same')(x)
    key = Convolution2D(filters=16, kernel_size=1, padding='same')(x)
    value = Convolution2D(filters=32, kernel_size=1, padding='same')(x)
    attn = Attention(32, 32)([query, value, key])
    x = Concatenate(axis=-1)([x, attn])

    x = Flatten()(x)
    x = Dense(2048, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))(x)
    output_layer = Dense(4, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    print(model.summary())

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# 训练分类器
# estimator = KerasClassifier(build_fn=baseline_model, epochs=300, batch_size=256, verbose=1)#当verbose=1时，带进度条的输出日志信息
# estimator.fit(X_train, Y_train)
# estimator.fit(X_train, Y_train, batch_size=256, epochs=300, verbose=1, validation_data=(X_test, Y_test), callbacks=[Metrics(valid_data=(X_test, Y_test))])

estimator = KerasClassifier(build_fn=baseline_model)

#if not os.path.exists('./checkpointsNN'):
#    os.makedirs('./checkpointsNN')

history = estimator.fit(X_train, Y_train, batch_size=256, epochs=1000, verbose=1, validation_data=(X_test, Y_test),
                        callbacks=[Metrics(valid_data=(X_test, Y_test)), checkpoint, early_stopping])

# 将其模型转换为json
save_model(estimator.model, 'model/my_model.h5')

#model_json = estimator.model.to_json()
#with open(r"model/model0401.json", 'w') as json_file:
#    json_file.write(model_json)  # 权重不在json中,只保存网络结构
#estimator.model.save_weights('model_eegnet_0401.h5')

print(history.history.keys())

plt.plot(history.history['accuracy'],'b--')
plt.plot(history.history['val_accuracy'],'y-')
plt.plot(history.history['val_f1'],'r-')
plt.plot(history.history['val_recall'],'g-')
plt.plot(history.history['val_precision'],'c-')

plt.title('model report')
plt.ylabel('evaluation')
plt.xlabel('epoch')
plt.legend(['train_accuracy', 'val_accuracy','val_f1-score','val_precisions','val_recalls'], loc='lower right')
#plt.savefig('results/result_acc.png')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuray')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc ='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('mode loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc = 'upper left')
plt.show()

# 加载模型用做预测
#json_file = open(r"model/model0401.json", "r")
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = model_from_json(loaded_model_json)
#loaded_model.load_weights('best_weights0401.h5')
#print("loaded model from disk")
#loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
loaded_model = keras.models.load_model('model/my_model.h5')

# 分类准确率
print("The accuracy of the classification model:")
scores = loaded_model.evaluate(X_test, Y_test, verbose=0)
print('%s: %.2f%%' % (loaded_model.metrics_names[1], scores[1] * 100))

test1 = pd.read_csv('OddballData0114/test.csv')
test2 = pd.read_csv('OddballData0114/test2.csv')
test = np.concatenate((test1, test2), axis = 0)
# print('test data', test.shape)
label_test = np.array(test[:, 0])
eeg_data_raw_test = np.array(test[:, 2:27].astype(float))
eeg_data_test = eeg_data_raw_test.transpose()
ch_names_test = ['CPz', 'Pz', 'Fz', 'FCz', 'Cz', 'C3', 'C1', 'CP3', 'P3', 'F3', 'FC3', 'C5', 'C4', 'C6', 'CP4', 'P4', 'F4', 'FC4', 'C2', 'PO6', 'O1', 'O2', 'PO5', 'PO3', 'PO4']
ch_types_test = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg']
sfreq_test = 1000  # Hz

info_test = mne.create_info(ch_names_test, sfreq_test, ch_types_test)
raw_test = mne.io.RawArray(eeg_data_test, info_test)
montage_test = mne.channels.make_standard_montage("standard_1020")
raw_test.set_montage(montage_test)

# 创建 events & event_id
events_test = np.zeros((1975, 3), dtype='int')
n = sfreq_test * 0
for m in range(1975):
    events_test[m, 0] = n
    n += sfreq_test * 1
    b = 1000*m
    events_test[m, 2] = label_test[b]
print('test events', events_test)
event_id_test = dict(non_workload = 0, low_workload = 1, high_workload = 2)

# 创建epochs
tmin, tmax = -0.0, 0.7  # 记录点的前0秒后0.8秒用于生成epoch数据
epochs_test = mne.Epochs(raw_test, events_test, event_id_test
                    , tmin, tmax
                    , proj=True
                    , baseline=(0, 0)
                    , preload=True
                    )

labels_test = epochs_test.events[:, -1]

#特征提取和分类
scores_test = []
epochs_data_test = epochs_test.get_data()
epochs_data_expend_test = np.expand_dims(epochs_data_test, axis=3)
X_valid = epochs_data_expend_test

encoder = LabelEncoder()
label_encoded_test = encoder.fit_transform(labels_test)
label_onehot_test = np_utils.to_categorical(label_encoded_test)
Y_valid = label_onehot_test

predictResult = loaded_model.predict(X_valid)
max_indices = np.argmax(predictResult, axis=1)
result = np.zeros_like(predictResult)
result[np.arange(predictResult.shape[0]), max_indices] = predictResult[np.arange(predictResult.shape[0]), max_indices]
print('result', result)
result = np.argmax(result, axis=1)
print('result', result)
np.savetxt('output/output.csv', result, delimiter=',', fmt='%d')