# %%
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.utils import shuffle
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import mplhep as hep
import os
import json
import logging

# %%

with open('settings.json') as json_file:
    settings = json.load(json_file)


train_on = settings["train_config"]["oddeven"]
dm = settings["train_config"]["dm"]
nbjet = settings["train_config"]["bjet"]
n_epochs = settings["train_config"]["n_epochs"]
batch_size = settings["train_config"]["batch_size"]
model_dir = settings["model_dir"]

# %%
now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, filename=f'E{train_on}_DM{dm}_bjet{nbjet}.log')
logging.info('Starting the script...')


logging.info(f"Train on: {train_on}")
logging.info(f"nbjet: {nbjet}")
logging.info(f"dm: {dm}")
logging.info(f"nbjet: {nbjet}")
logging.info(f"n_epochs: {n_epochs}")
logging.info(f"batch_size: {batch_size}")

# %%
delta_m = dm
signal_dfs = []
for dm in delta_m:
    signal_file = settings["samples"][f"dm{dm}"]
    logging.info(f"Signal file: {signal_file}")
    signal_df = pd.read_csv(f'{signal_file}')
    signal_dfs.append(signal_df)
signal_df = pd.concat(signal_dfs)
signal_df['label'] = 1
print(signal_df.shape)

# %%
backgrounds = [settings["samples"]["ttbar"], settings["samples"]['VV']]
background_dfs = {k: pd.read_csv(f'{k}') for k in backgrounds}
print(background_dfs)

# %%
ttbar_df = background_dfs[backgrounds[0]].copy()
VV_df = background_dfs[backgrounds[1]].copy()

# %%
# Do more tighter cuts on MDR
#mdr_tight = 75
#signal_df = signal_df[(signal_df['MDR'] > mdr_tight)]
#ttbar_df = ttbar_df[(ttbar_df['MDR'] > mdr_tight)]
#VV_df = VV_df[(VV_df['MDR'] > mdr_tight)]
#Zjets_df = Zjets_df[(Zjets_df['MDR'] > mdr_tight)]

# %%
# MET over ptlep
#signal_df["met_over_ptlep"] = signal_df["MET"] / (signal_df["lep1pT"] + signal_df["lep2pT"])
#ttbar_df["met_over_ptlep"] = ttbar_df["MET"] / (ttbar_df["lep1pT"] + ttbar_df["lep2pT"])
#Zjets_df["met_over_ptlep"] = Zjets_df["MET"] / (Zjets_df["lep1pT"] + Zjets_df["lep2pT"])
#VV_df["met_over_ptlep"] = VV_df["MET"] / (VV_df["lep1pT"] + VV_df["lep2pT"])

# %%
"""density = True
signal_df["met_over_ptlep"].plot.hist(bins=15, range=(0, 15), histtype='step', label='Signal DM 90', color='cyan', density=density, linestyle='--')
ttbar_df["met_over_ptlep"].plot.hist(bins=15, range=(0, 15), histtype='step', label='ttbar', color='yellow', density=density)
#Zjets_df["met_over_ptlep"].plot.hist(bins=15, range=(0, 15), histtype='step', label='Zjets', color='green', density=density)
VV_df["met_over_ptlep"].plot.hist(bins=15, range=(0, 15), histtype='step', label='VV', color='purple', density=density)
plt.xlabel('MET / (lep1pT + lep2pT)')
plt.ylabel('Density')
plt.legend()
plt.savefig('met_over_ptlep.pdf', dpi=300)"""

# %%
if train_on == 'even':
    signal_df = signal_df.query('EventNumber % 2 == 0')
    ttbar_df = ttbar_df.query('EventNumber % 2 == 0')
    VV_df = VV_df.query('EventNumber % 2 == 0')
    #Zjets_df = Zjets_df.query('EventNumber % 2 == 0')

elif train_on == 'odd':
    signal_df = signal_df.query('EventNumber % 2 == 1')
    ttbar_df = ttbar_df.query('EventNumber % 2 == 1')
    VV_df = VV_df.query('EventNumber % 2 == 1')
    #Zjets_df = Zjets_df.query('EventNumber % 2 == 1')

else:
    raise ValueError('train_on must be even or odd')
    
print('signal', signal_df.shape, 'ttbar', ttbar_df.shape, 'VV', VV_df.shape)
logging.info(f'signal {signal_df.shape}, ttbar {ttbar_df.shape}, VV {VV_df.shape}')
print('Training on ', train_on)

# %%
ttbar_df['label'] = 2
VV_df['label'] = 3
#Zjets_df['label'] = 4

# %%
weights = ["WeightEventselSF", "WeightEventsmuSF", "WeightEventsJVT", "WeightEventsbTag", "WeightEventsPU", "WeightEventsSF_global"]

#features_0b = ["lep1pT", "lep1eta", "lep2pT", "lep2eta", "MDR", "MET", "METsig", "mll", "isSF", "RPT", "gamInvRp1", "DPB_vSS", "cosTheta_b", "nbjet"]
features_0b = ["lep1pT", "lep1eta", "lep2pT", "lep2eta",  "MDR", "MET", "METsig", "mll", "dPhil1l2", "isSF", "RPT", "gamInvRp1", "DPB_vSS",  "nbjet"]
features_1b = features_0b + ["bjet1pT"]
features_2b = features_1b + ["bjet2pT"]

nbjet_features = {
    0: features_0b,
    1: features_1b,
}
necessary_vars = ['EventNumber', 'label']
model_features = nbjet_features[nbjet] + necessary_vars + weights
logging.info(f"Model features: {model_features}")

# %%
signal_df = signal_df[model_features]
ttbar_df = ttbar_df[model_features]
VV_df = VV_df[model_features]
#Zjets_df = Zjets_df[model_features]

# %%
#Applying weights
def add_weights(df, is_signal=False):
    if is_signal:
        df.loc[:, 'weight'] = df['WeightEventselSF']*df['WeightEventsmuSF']*df['WeightEventsJVT']*df['WeightEventsbTag']*df['WeightEventsPU']*df['WeightEventsSF_global']
    else:
        df.loc[:, 'weight'] = df['WeightEventselSF']*df['WeightEventsmuSF']*df['WeightEventsJVT']*df['WeightEventsbTag']*df['WeightEventsPU']*df['WeightEventsSF_global']
    df.drop(columns=['WeightEventselSF', 'WeightEventsmuSF', 'WeightEventsJVT', 'WeightEventsbTag', 'WeightEventsPU', 'WeightEventsSF_global'], inplace=True)
    return df

# %%
signal_df = add_weights(signal_df, is_signal=True)
ttbar_df = add_weights(ttbar_df)
VV_df = add_weights(VV_df)
#Zjets_df = add_weights(Zjets_df)

# %%
#filter events with nbjet number
if  nbjet > 0:
    sel_string = 'nbjet > 0'
else:
    sel_string = 'nbjet == 0'

print('Applying cut', sel_string)
signal_df = signal_df.query(sel_string)
ttbar_df = ttbar_df.query(sel_string)
VV_df = VV_df.query(sel_string)
#Zjets_df = Zjets_df.query(sel_string)

features = signal_df.columns
scale_features = signal_df.columns[:-4]

print('Signal', signal_df.shape, 'ttbar', ttbar_df.shape, 'VV', VV_df.shape)

# %%
import math
N = len(signal_df) + len(ttbar_df) + len(VV_df)
n = 3
class_weights = {
    1: N / (len(signal_df) * n),
    2: N / (len(ttbar_df) * n),
    3: N / (len(VV_df) * n),
    #4: N / (len(Zjets_df) * n)
}
print('Original class weights', class_weights)
logging.info(f'Original class weights {class_weights}')


# %%
def add_class_weights(df):
    df.loc[:, 'class_weight'] = df['label'].map(class_weights)
    df['weight'] = df['weight']*df['class_weight']
    df.drop(columns=['class_weight'], inplace=True)
    return df

# %%
signal_df = add_class_weights(signal_df)
ttbar_df = add_class_weights(ttbar_df)
VV_df = add_class_weights(VV_df)
logging.info('class weights added')
#Zjets_df = add_class_weights(Zjets_df)

# %%
#Number of events in each bjet category
print('signal bjet - 0 ', len(signal_df.query('nbjet == 0')), '1 ', len(signal_df.query('nbjet == 1')), '2 ', len(signal_df.query('nbjet == 2')))
print('ttbar bjet - 0 ', len(ttbar_df.query('nbjet == 0')), '1 ', len(ttbar_df.query('nbjet == 1')), '2 ', len(ttbar_df.query('nbjet == 2')))
print('VV bjet - 0 ', len(VV_df.query('nbjet == 0')), '1 ', len(VV_df.query('nbjet == 1')), '2 ', len(VV_df.query('nbjet == 2')))

logging.info(f'signal bjet - 0: {len(signal_df.query("nbjet == 0"))}, '
             f'1: {len(signal_df.query("nbjet == 1"))}, '
             f'2: {len(signal_df.query("nbjet == 2"))}')

logging.info(f'ttbar bjet - 0: {len(ttbar_df.query("nbjet == 0"))}, '
             f'1: {len(ttbar_df.query("nbjet == 1"))}, '
             f'2: {len(ttbar_df.query("nbjet == 2"))}')

logging.info(f'VV bjet - 0: {len(VV_df.query("nbjet == 0"))}, '
             f'1: {len(VV_df.query("nbjet == 1"))}, '
             f'2: {len(VV_df.query("nbjet == 2"))}')

# %%
input_df = pd.concat([signal_df, ttbar_df, VV_df])
input_df = shuffle(input_df)
input_df = input_df.astype('float32')
logging.info(f'input_df made.  shape {input_df.shape}')

# %%
X = input_df[scale_features].copy()
#X = input_df[scale_features + 'weight'].copy()
#X.drop(columns=['weight'], inplace=True)
Z = input_df['weight']
X = X.astype('float32')
Y = input_df['label']
le = LabelEncoder()
Y = le.fit_transform(Y)
Y = to_categorical(Y)

X_train, X_TEMP, Y_train, Y_TEMP, Z_Train, Z_TEMP = train_test_split(X, Y,  Z, test_size=0.2, shuffle=True, random_state=42, stratify=Y)
X_val, X_test, Y_val, Y_test, Z_val, Z_test = train_test_split(X_TEMP, Y_TEMP, Z_TEMP, test_size=0.5, shuffle=True, random_state=42, stratify=Y_TEMP)

#min max scaling
#columns = nbjet_features[nbjet]
columns = scale_features
min_values = X_train[columns].min()
max_values = X_train[columns].max()
X_train = (X_train - min_values) / (max_values - min_values)    
X_val = (X_val - min_values) / (max_values - min_values)
X_test = (X_test - min_values) / (max_values - min_values)

if dm == 120: tag = "small"
else: tag = "large"

with open(f'scaler_{tag}DM_bjet{nbjet}_{train_on}.txt', 'w') as f:
        for i, var in enumerate(columns):
            f.write(f"{var}\t{min_values.values[i]}\t{max_values.values[i]}\n")
        print(f"Scaler saved to scaler{dm}_{train_on}_{now}.txt")
        
print('X_train', X_train.shape, 'Y_train', Y_train.shape, 'Z_Train', Z_Train.shape)
print('X_val', X_val.shape, 'Y_val', Y_val.shape, 'Z_val', Z_val.shape)
print('X_test', X_test.shape, 'Y_test', Y_test.shape, 'Z_test', Z_test.shape)
print('Features', X_train.columns)

logging.info(f'X_train shape {X_train.shape}, Y_train shape {Y_train.shape}, Z_Train shape {Z_Train.shape}')
logging.info(f'X_val shape {X_val.shape}, Y_val shape {Y_val.shape}, Z_val shape {Z_val.shape}')
logging.info(f'X_test shape {X_test.shape}, Y_test shape {Y_test.shape}, Z_test shape {Z_test.shape}')
logging.info(f'Features {X_train.columns}')
logging.info(f'Scaler saved to scaler{dm}_{train_on}_{now}.txt')

# %%
Z_Train = Z_Train.to_numpy()
Z_val = Z_val.to_numpy()

# %%
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Input, Activation
import tensorflow as tf
from keras.optimizers import Adam
alpha= class_weights
loss = 'categorical_crossentropy'
metrics=[
        tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(multi_label=False, name="AUC")
    ]

def dnn_model(num_features, num_classes):
    optimizer = Adam(learning_rate=0.0001)
    model = Sequential()
    model.add(Input(shape=(num_features,)))
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics, weighted_metrics=[keras.metrics.CategoricalAccuracy(name='accuracy')])
    return model

# %%
model = dnn_model(X_train.shape[1], Y_train.shape[1])
model.summary()

# %%
Y_train

# %%
history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val, Z_val), epochs=n_epochs, batch_size=batch_size, verbose=1, sample_weight=Z_Train)

#shap - begin
import shap
signal_idx = np.where(Y_train[:,0] == 1)[0]
ttbar_idx = np.where(Y_train[:,1] == 1)[0]
vv_idx = np.where(Y_train[:,2] == 1)[0]
n_samples_per_class = min(len(signal_idx), len(ttbar_idx), len(vv_idx), 200)

signal_sample = np.random.choice(signal_idx, n_samples_per_class, replace=False)
ttbar_sample = np.random.choice(ttbar_idx, n_samples_per_class, replace=False)
vv_sample = np.random.choice(vv_idx, n_samples_per_class, replace=False)

balanced_indices = np.concatenate([signal_sample, ttbar_sample, vv_sample])
shap_sample = X_train.iloc[balanced_indices].to_numpy()

explainer = shap.DeepExplainer(model, shap_sample)
shap_values = explainer.shap_values(shap_sample)

def shap_values_to_list(shap_values, model):
    shap_as_list=[]
    for i in range(3):
        shap_as_list.append(shap_values[:,:,i])
    return shap_as_list
shap_as_list = shap_values_to_list(shap_values, model)
plt.clf()
shap.summary_plot(shap_as_list, shap_sample, plot_type="bar", feature_names=X_train.columns, show=False)
plt.savefig(f'shap_imp_{nbjet}.pdf', dpi=300)
#shap - end


# %%
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model.save(f'{model_dir}/model_{train_on}_{nbjet}bjets_dm{dm}_{now}.h5')
logging.info(f'{model_dir}/model_{train_on}_{nbjet}bjets_dm{dm}_{now}.h5 saved')

# %%
plot_dir = settings["plot_dir"]
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
accuracy = history.history['accuracy']  
val_accuracy = history.history['val_accuracy']

min_accuracy = min(min(val_accuracy), min(accuracy))
max_accuracy = max(max(val_accuracy), max(accuracy))

fig, ax = plt.subplots(2,2, figsize=(10,5))
ax[0,0].plot(epochs, accuracy, label='Training accuracy')
ax[0,0].plot(epochs, val_accuracy, label='Validation accuracy')
ax[0,0].set_ylabel('Accuracy')
ax[0,0].set_ylim([min_accuracy - 0.1, max_accuracy + 0.1])

fig.tight_layout(pad=3.0)

ax[0, 1].plot(epochs, history.history["weighted_accuracy"], label='Training')
ax[0, 1].plot(epochs, history.history["val_weighted_accuracy"], label='Validation')
ax[0, 1].set_ylabel('Weighted Accuracy')
ax[0,1].set_ylim([min_accuracy - 0.1, max_accuracy + 0.1])

ax[1,0].plot(epochs, loss, label='Training')
ax[1,0].plot(epochs, val_loss, label='Validation ')
ax[1,0].set_ylabel('Loss')

ax[1,1].plot(epochs, history.history["AUC"], label='Training')
ax[1,1].plot(epochs, history.history["val_AUC"], label='Validation')
ax[1,1].set_ylabel('AUC')
plt.legend()
fig.supxlabel('Epochs')
plt.savefig(f'{plot_dir}/acc_loss_auc_DM_{dm}_nbjet_{nbjet}_{train_on}_{now}.pdf')
logging.info(f'Plot saved to {plot_dir}/acc_loss_auc_DM_{dm}_nbjet_{nbjet}_{train_on}_{now}.pdf')

# %%
Y_pred = model.predict(X_test)
sig_sig_score = []
sig_ttbar_score = []
sig_VV_score = []
sig_Zjets_score = []
ttbar_sig_score = []
ttbar_ttbar_score = []
ttbar_VV_score = []
ttbar_Zjets_score = []
VV_sig_score = []
VV_ttbar_score = []
VV_VV_score = []
VV_Zjets_score = []
Zjets_sig_score = []
Zjets_ttbar_score = []
Zjets_VV_score = []
Zjets_Zjets_score = []
for event, label in enumerate(Y_test):
    if label[0] == 1:
        sig_sig_score.append(Y_pred[event][0])
        sig_ttbar_score.append(Y_pred[event][1])
        sig_VV_score.append(Y_pred[event][2])
        #sig_Zjets_score.append(Y_pred[event][2])
    elif label[1] == 1:
        ttbar_sig_score.append(Y_pred[event][0])
        ttbar_ttbar_score.append(Y_pred[event][1])
        ttbar_VV_score.append(Y_pred[event][2])
        #ttbar_Zjets_score.append(Y_pred[event][2])
    elif label[2] == 1:
        VV_sig_score.append(Y_pred[event][0])
        VV_ttbar_score.append(Y_pred[event][1])
        VV_VV_score.append(Y_pred[event][2])
        #VV_Zjets_score.append(Y_pred[event][3])
    #elif label[2] == 1:
    #    Zjets_sig_score.append(Y_pred[event][0])
    #    Zjets_ttbar_score.append(Y_pred[event][1])
    #    Zjets_VV_score.append(Y_pred[event][2])
    #    Zjets_Zjets_score.append(Y_pred[event][2])

# %%
#combine all scores
fig, ax = plt.subplots(3, 1, figsize=(10, 8))
ax[0].hist(sig_sig_score, bins=50, alpha=0.5, label='Signal as Signal')
ax[0].hist(sig_ttbar_score, bins=50, alpha=0.5, label='Signal as ttbar')
ax[0].hist(sig_VV_score, bins=50, alpha=0.5, label='Signal as VV')
ax[0].legend()
ax[0].set_title('Signal')
ax[0].set_xlabel('Model Output')
ax[0].set_ylabel('Counts')
ax[0].set_xlim(0,1)

ax[1].hist(ttbar_sig_score, bins=50, alpha=0.5, label='ttbar as Signal')
ax[1].hist(ttbar_ttbar_score, bins=50, alpha=0.5, label='ttbar as ttbar')
ax[1].hist(ttbar_VV_score, bins=50, alpha=0.5, label='ttbar as VV')
ax[1].legend()
ax[1].set_title('ttbar')
ax[1].set_xlabel('Model Output')
ax[1].set_ylabel('Counts')
ax[1].set_xlim(0,1)

ax[2].hist(VV_sig_score, bins=50, alpha=0.5, label='VV as Signal')
ax[2].hist(VV_ttbar_score, bins=50, alpha=0.5, label='VV as ttbar')
ax[2].hist(VV_VV_score, bins=50, alpha=0.5, label='VV as VV')
ax[2].legend()
ax[2].set_title('VV')
ax[2].set_xlabel('Model Output')
ax[2].set_ylabel('Counts')
plt.tight_layout()
ax[2].set_xlim(0,1)

#for i in [0, 1, 2]:
#    ax[i].set_yscale('log')

fig.savefig(f'{plot_dir}/all_scores_DM_{dm}_nbjet_{nbjet}_{train_on}_{now}.pdf')
logging.info(f'Plot saved to {plot_dir}/all_scores_DM_{dm}_nbjet_{nbjet}_{train_on}_{now}.pdf')

# %%
print('Analysis Finished')
logging.info('Analysis Finished')
