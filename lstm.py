
import pandas as pd 
import numpy as np 
import os 
import tensorflow as tf 
from tensorflow.keras.layers import LSTM,Dropout,Dense 
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns 

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
from collections import Counter
import argparse 
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument('--add-static',default=False, help='add static patient data or not: True/False')
parser.add_argument('--add-meds', default=False)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--raw-data-dir', default='./raw_data')
parser.add_argument('--processed-data-dir',default='./processed_data')
args = parser.parse_args()


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

def split_train_test(df, frac=0.2):
    
    df = df.drop_duplicates(subset = 'id')
    msk = np.random.rand(len(df)) < frac
    test_ids = df[msk]['id']
    train_ids = df[~msk]['id']
    
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    
    for patient_id, group in data.groupby('id'):
        sequence_features = group[FEATURE_COLUMNS]
        label = labels[labels.id == patient_id].iloc[0].Stage_Progress
        if patient_id in test_ids.tolist():
            X_test.append(sequence_features.values)
            y_test.append(label)
        else:
            X_train.append(sequence_features.values)
            y_train.append(label)
    
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)



# assert(False)
#get the static data 
patients = pd.read_csv('raw_data/T_demo.csv')
#transform the categorical values into discrete values 
patients['gender'] = np.where(patients['gender']=='Female',1,0)
# convert categorical columns into binary values 
patients = pd.concat([patients.drop('race',axis=1),pd.get_dummies(patients['race'])],axis=1)
patients['age'] = scaler.fit_transform(patients['age'].values.reshape(-1,1))

patients_arr = np.array(patients.drop('id',axis=1).values)


def prep_data(add_meds=False):
    print(f'add meds: {args.add_meds}')
    if args.add_meds:
        print(f'i am in the if {args.add_meds}')
        cmd = ['python', 'data_preprocess.py', '--add-meds=True']
        subprocess.run(cmd)
    else:
        subprocess.run(['python','data_preprocess.py']) 
    data = pd.read_csv('processed_data/data.csv')
    return data 

data = prep_data(args.add_meds)  
labels = pd.read_csv('raw_data/T_stage.csv')
labels.Stage_Progress = labels.Stage_Progress.replace({True:1, False:0})

FEATURE_COLUMNS = data.columns.tolist()[2:]
inputs =[]
targets =[]
for patient_id, group in data.groupby('id'):
    sequence_features = group[FEATURE_COLUMNS]
    label = labels[labels.id == patient_id].iloc[0].Stage_Progress
    inputs.append(sequence_features.values)
    targets.append(label)

inputs= np.array(inputs)
targets = np.array(targets)

sampling=10
num_folds=5


# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)
acc_per_fold=[]
loss_per_fold=[]
prec_per_fold =[]
rec_per_fold=[]

actual_classes = []
predicted_classes = []
# K-fold Cross Validation model evaluation
fold_no = 1

def lstm_default():
    input_1 = tf.keras.Input(shape = (700, len(FEATURE_COLUMNS)))
    average_pooling_1 = tf.keras.layers.AveragePooling1D(sampling, strides=2, data_format='channels_last')(input_1)

    lstm_1 = LSTM(64, return_sequences=True, dropout=0.2, input_shape = (700//sampling, len(FEATURE_COLUMNS)))(average_pooling_1)
    layer_norm_1 = tf.keras.layers.LayerNormalization(axis=2)(lstm_1)
    average_pooling_2 = tf.keras.layers.AveragePooling1D(2, data_format='channels_last')(layer_norm_1)
    lstm_2 = LSTM(32, dropout=0.2)(average_pooling_2)
    dropout_1 = tf.keras.layers.Dropout(0.5)(lstm_2)
    out = Dense(1, activation=None)(dropout_1)
    model = tf.keras.models.Model(inputs=input_1, outputs = out)

    return model 


def lstm_static():
    input_1 = tf.keras.Input(shape = (700, len(FEATURE_COLUMNS)))
    input_2 = tf.keras.Input(shape = (7,))
    average_pooling_1 = tf.keras.layers.AveragePooling1D(sampling, strides=2, data_format='channels_last')(input_1)
    # lstm_1 = LSTM(64, return_sequences=True, dropout=0.2, input_shape = (700//sampling, len(FEATURE_COLUMNS)))(input_1)

    lstm_1 = LSTM(64, return_sequences=True, dropout=0.2, input_shape = (700//sampling, len(FEATURE_COLUMNS)))(average_pooling_1)
    layer_norm_1 = tf.keras.layers.LayerNormalization(axis=2)(lstm_1)
    average_pooling_2 = tf.keras.layers.AveragePooling1D(2, data_format='channels_last')(layer_norm_1)
    lstm_2 = LSTM(32, dropout=0.2)(average_pooling_2)
    # lstm_2 = LSTM(32, dropout=0.2)(lstm_1)
    concatted = tf.keras.layers.Concatenate(axis=1)([lstm_2, input_2])
    dropout_1 = tf.keras.layers.Dropout(0.5)(concatted)
    out = Dense(1, activation=None)(dropout_1)
    
    # out = Dense(1, activation=None)(concatted)
    
    model = tf.keras.models.Model(inputs=[input_1, input_2], outputs = out)

    return model 
class_1_results = []
for train_index, test_index in kfold.split(inputs, targets):
    # for tra_index, val_index in kfold.split(train_index):
    #     print(tra_index, val_index)
    print(f"selecting the model: {args.add_static}")
    if args.add_static:
        X_train = [inputs[train_index],patients_arr[train_index]]
        X_test = [inputs[test_index],patients_arr[test_index]]
        model = lstm_static()
    else:
        X_train = inputs[train_index]
        X_test = inputs[test_index]
        model = lstm_default()

    y_train = targets[train_index]
    y_test =  targets[test_index]  

    actual_classes.append(y_test)

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0),
                         tf.keras.metrics.Precision(thresholds=0),
                         tf.keras.metrics.Recall(thresholds=0)],
                optimizer=tf.keras.optimizers.experimental.AdamW(learning_rate=0.001, weight_decay=0.0001))
    callbacks = [
        ModelCheckpoint(
          "./model/best_model.h5", save_best_only=True, monitor="loss"
      ),
      # ReduceLROnPlateau(
      #     monitor="loss", factor=0.5, patience=20, min_lr=0.0001
      # ),
      EarlyStopping(monitor="loss", patience=5, verbose=1),
    ]

    # model.build()
    model.summary()
   # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    class_weight = {0: 1., 1:2.}

    # Fit data to model  
    model.fit(X_train, y_train, 
               batch_size=64, 
               epochs=100, 
               class_weight=class_weight,
            #    validation_split=0.10,    
               callbacks=[callbacks])

    # print("Loading model")
    # model = load_model('best_model.h5')

    # print(f"train performance {model.evaluate(X_train, y_train)}")
    print(f"train value counts {Counter(y_test)}")
    print(f"test value counts {Counter(y_test)}")
    results = model.evaluate(X_test, y_test, batch_size=64)
    print("test loss, test acc:", results)

    y_test_prob = model.predict(X_test, verbose=1)  
    y_test_pred = np.where(y_test_prob > 0.5, 1, 0)
    predicted_classes.append( y_test_pred)


    print(confusion_matrix(y_test,y_test_pred))
    print(classification_report(y_test,y_test_pred,output_dict=True).keys())
    class_1_results.append(classification_report(y_test,y_test_pred, output_dict=True)['1'])
    acc_per_fold.append(results[1])
    loss_per_fold.append(results[0])
    prec_per_fold.append(results[2])
    rec_per_fold.append(results[3])
    fold_no +=1

print(class_1_results)
print(f"average accuracy cross-validation {np.mean(acc_per_fold)}")
print(f"average loss cross-validation {np.mean(loss_per_fold)}")
print(f"average precision cross-validation {np.mean(prec_per_fold)}")
print(f"average recall cross-validation {np.mean(rec_per_fold)}")

