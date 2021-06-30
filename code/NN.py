import numpy as np 
import pandas as pd
import os

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers

pwd = os.getcwd()

train = pd.read_csv(pwd+"/data/train.csv")
test = pd.read_csv(pwd+"/data/test.csv")
targets = pd.get_dummies(train['target'])

cce = tf.keras.losses.CategoricalCrossentropy()

es = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=1e-05, patience=8, verbose=0,
    mode='min', baseline=None, restore_best_weights=True)

plateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.7, patience=2, verbose=0)

def get_model():
    inputs = layers.Input(shape = (75,))
    
    embed = layers.Embedding(360, 8)(inputs)
    embed = layers.Flatten()(embed)
    
    hidden = layers.Dropout(0.2)(embed)
    hidden = tfa.layers.WeightNormalization(layers.Dense(units=32, activation='selu', kernel_initializer="lecun_normal"))(hidden)
    
    output = layers.Dropout(0.2)(layers.Concatenate()([embed, hidden]))
    output = tfa.layers.WeightNormalization(layers.Dense(units=32, activation='relu'))(output) 
    
    output = layers.Dropout(0.3)(layers.Concatenate()([embed, hidden, output]))
    output = tfa.layers.WeightNormalization(layers.Dense(units=32, activation='elu'))(output) 
    output = layers.Dense(9, activation = 'softmax')(output)
    
    model = keras.Model(inputs=inputs, outputs=output, name="res_nn_model")
    
    return model

def custom_metric(y_true, y_pred):
    y_pred = K.clip(y_pred, 1e-15, 1-1e-15)
    loss = K.mean(cce(y_true, y_pred))
    return loss

iter = 0

while(1):  #不停训练，选取最好结果
    iter = iter + 1
    print(f"the {iter} times iteration")

    NN_a_train_preds = []
    NN_a_test_preds  = []
    NN_a_oof_pred3 = []

    oof_NN_a = np.zeros((train.shape[0], 9))
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=111)

    for fold, (tr_idx, ts_idx) in enumerate(skf.split(train, train.iloc[:,-1])):
        X_train = train.iloc[:,1:-1].iloc[tr_idx]
        y_train = targets.iloc[tr_idx]
        X_test = train.iloc[:,1:-1].iloc[ts_idx]
        y_test = targets.iloc[ts_idx]
        K.clear_session()
    
        model_attention = get_model()

        model_attention.compile(loss='categorical_crossentropy',
                                optimizer = keras.optimizers.Adam(learning_rate=0.0002),
                                metrics=custom_metric)
    
        model_attention.fit(X_train, y_train,
                            batch_size = 256, 
                            epochs = 100,
                            validation_data=(X_test, y_test),
                            callbacks=[es, plateau],
                            verbose = 0)
    
        pred_a = model_attention.predict(X_test) 
        oof_NN_a[ts_idx] += pred_a 
        score_NN_a = log_loss(y_test, pred_a)
        print(f"\nFOLD {fold} Score NN Attention model: {score_NN_a}")
    
        NN_a_test_preds.append(model_attention.predict(test.iloc[:,1:]))
    
    score_a = log_loss(targets, oof_NN_a)
    print('=' * 40)
    print(f"\niter: {iter}== FINAL SCORE: {score_a} =====\n")
    print('=' * 40)

    temp = 1.750
    if score_a < temp:
        proba = sum(np.array(NN_a_test_preds)/10)
        output = pd.DataFrame({'id': test['id'], 'Class_1': proba[:, 0], 'Class_2': proba[:, 1], 'Class_3': proba[:, 2],
                               'Class_4': proba[:, 3], 'Class_5': proba[:, 4], 'Class_6': proba[:, 5],
                               'Class_7': proba[:, 6], 'Class_8': proba[:, 7], 'Class_9': proba[:, 8]})

        output.to_csv('NN.csv', index=False)
        temp = score_a