import numpy as np
import pandas as pd
import glob
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from eegmodels import EEGNet, DeepConvNet, ShallowConvNet
import sys
import time
from tensorflow.keras.callbacks import Callback
from scipy.signal import butter, filtfilt
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from collections import defaultdict
import matplotlib.pyplot as plt


#Training Process Bar Callback for Batch and Epoch Metrics
class TrainingProgressBar(Callback):
    def __init__(self, model_name, total_epochs, total_batches):
        super().__init__()
        self.model_name = model_name
        self.total_epochs = total_epochs
        self.total_batches = total_batches

    def on_train_begin(self, logs=None):
        print(f"\n===== Training {self.model_name} =====\n")

    def on_epoch_begin(self, epoch, logs=None):
        print(f"\nEpoch {epoch+1}/{self.total_epochs}")

    def on_train_batch_end(self, batch, logs=None):
        bar_length = 30
        filled = int((batch+1) / self.total_batches * bar_length)
        bar = "=" * filled + "-" * (bar_length - filled)
        sys.stdout.write(
            f"\r[{bar}] Batch {batch+1}/{self.total_batches} "
            f"- loss: {logs.get('loss'):.4f} - acc: {logs.get('accuracy'):.4f}"
        )
        sys.stdout.flush()

    def on_epoch_end(self, epoch, logs=None):
        print(
            f"\nEpoch {epoch+1} completed. "
            f"val_loss: {logs.get('val_loss'):.4f}, val_acc: {logs.get('val_accuracy'):.4f}"
        )


#Plot Training and Validation Curves for a Model
def plot_training_curves(history, model_name):
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


#Apply bandpass Filter to EEG Data
def bandpass_filter(data, low=1, high=40, fs=200):
    b, a = butter(4, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, data, axis=2)


#Focal Loss Function for Imbalanced Classification
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1-1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        loss = alpha * tf.math.pow(1 - y_pred, gamma) * cross_entropy
        return tf.reduce_sum(loss, axis=1)
    return focal_loss_fixed


#Add Gaussian Noise to Input Data
def add_gaussian_noise(X, std=0.01):
    noise = np.random.normal(0, std, X.shape).astype(np.float32)
    return X + noise


#Balanced Batch Generator with Optional Data Augmentation
def balanced_batch_generator(X, y, batch_size=16, augment=True):
    y_int = y.argmax(axis=1)
    class_idx = defaultdict(list)
    for i, label in enumerate(y_int):
        class_idx[label].append(i)
    classes = list(class_idx.keys())
    while True:
        X_batch = []
        y_batch = []
        n_per_class = batch_size // len(classes)
        for cls in classes:
            idx = np.random.choice(class_idx[cls], n_per_class, replace=True)
            X_batch.append(X[idx])
            y_batch.append(y[idx])
        X_batch = np.concatenate(X_batch, axis=0)
        y_batch = np.concatenate(y_batch, axis=0)
        if augment:
            X_batch = add_gaussian_noise(X_batch, std=0.01)
        yield X_batch, y_batch


#Load and Format EEG Trials from CSV Files
def load_trials(path_pattern, num_trials_list, target_samples=128):
    files = sorted(glob.glob(path_pattern))
    all_trials = []

    for file_idx, f in enumerate(files):
        print(f"Loading {f}")
        df = pd.read_csv(f, header=0, sep=',')
        df = df.iloc[:, 1:57]
        arr = df.values.astype(np.float32)

        num_trials = num_trials_list[file_idx]
        trial_length = arr.shape[0] // num_trials

        for t in range(num_trials):
            start = t * trial_length
            end = start + target_samples
            if end > arr.shape[0]:
                end = arr.shape[0]
                start = end - target_samples
            trial_data = arr[start:end, :]
            trial_data = trial_data.T
            all_trials.append(trial_data[:, :target_samples])

    all_trials = np.stack(all_trials)
    all_trials = all_trials[..., np.newaxis]
    return all_trials


#Load Labels from CSV Files
def load_labels(label_file):
    labels = pd.read_csv(label_file, sep=';')
    labels = labels['Prediction'].values
    return labels

#Load Training, Validation, and Test Labels
y_train_raw = load_labels('dataset/ern/TrainLabels.csv')
y_val_raw = load_labels('dataset/ern/ValLabels.csv')
y_test_raw = load_labels('dataset/ern/TestLabels.csv')

num_classes = len(np.unique(y_train_raw))

#Convert Labels to One-Hot Encoded Format
y_train = to_categorical(y_train_raw, num_classes)
y_val = to_categorical(y_val_raw, num_classes)
y_test = to_categorical(y_test_raw, num_classes)

print("Load Training Dataset:")
num_trials_train = [60, 60, 60, 60, 100]*12  # 5 session * 12 alany
X_train = load_trials('dataset/ern/train/Data_S*_Sess*.csv', num_trials_train, target_samples=128)
print(X_train.shape)

print("Load Validation Dataset:")
num_trials_val = [60, 60, 60, 60, 100]*2
X_val = load_trials('dataset/ern/validation/Data_S*_Sess*.csv', num_trials_val, target_samples=128)
print(X_val.shape)

print("Load Test Dataset:")
num_trials_test = [60, 60, 60, 60, 100]*2
X_test = load_trials('dataset/ern/test/Data_S*_Sess*.csv', num_trials_test, target_samples=128)
print(X_test.shape)

print("Applying bandpass filter 1–40 Hz...")
X_train = bandpass_filter(X_train)
X_val   = bandpass_filter(X_val)
X_test  = bandpass_filter(X_test)

print("Normalizing channels…")
mean = X_train.mean(axis=(0, 2), keepdims=True)
std  = X_train.std(axis=(0, 2), keepdims=True)

X_train = (X_train - mean) / (std + 1e-6)
X_val   = (X_val - mean)   / (std + 1e-6)
X_test  = (X_test - mean)  / (std + 1e-6)

#Set Model and Training Parameters
Chans = 56
Samples = 128
nb_classes = num_classes
batch_size = 16
total_batches = len(X_train) // batch_size
epochs = 200
lr_schedule = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=10, min_lr=1e-5, verbose=1)

# Initialize and Compile EEG Classification Models
model1 = EEGNet(nb_classes=nb_classes, Chans=Chans, Samples=Samples, F1=16, D=2, F2=32, dropoutRate=0.5, dropoutType='Dropout')
model1.compile(optimizer=Adam(learning_rate=1e-3), loss=focal_loss(gamma=2, alpha=0.25), metrics=['accuracy'])

model2 = DeepConvNet(nb_classes=nb_classes, Chans=Chans, Samples=Samples)
model2.compile(optimizer=Adam(learning_rate=1e-3), loss=focal_loss(gamma=2, alpha=0.25), metrics=['accuracy'])

model3 = ShallowConvNet(nb_classes=nb_classes, Chans=Chans, Samples=Samples)
model3.compile(optimizer=Adam(learning_rate=1e-3), loss=focal_loss(gamma=2, alpha=0.25), metrics=['accuracy'])

#Define Callbacks for Model Training
callbacks_eeg = [TrainingProgressBar("EEGNet", epochs, total_batches), EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True), ModelCheckpoint('best_eegnet.h5', save_best_only=True, monitor='val_accuracy'), lr_schedule]
callbacks_deep = [TrainingProgressBar("DeepConvNet", epochs, total_batches), EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True), ModelCheckpoint('best_deepconvnet.h5', save_best_only=True, monitor='val_accuracy'), lr_schedule]
callbacks_shallow = [TrainingProgressBar("ShallowConvNet", epochs, total_batches), EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True), ModelCheckpoint('best_shallowconvnet.h5', save_best_only=True, monitor='val_accuracy'), lr_schedule]

train_gen = balanced_batch_generator(X_train, y_train, batch_size=batch_size, augment=True)
steps_per_epoch = len(X_train) // batch_size

#Train EEGNet, DeeConvNet, and ShallowConvNet Models
eegnet = model1.fit(train_gen, steps_per_epoch=steps_per_epoch, epochs=epochs, validation_data=(X_val, y_val), callbacks=callbacks_eeg)
deepconvnet = model2.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), callbacks=callbacks_deep)
shallowconvnet = model3.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), callbacks=callbacks_shallow)

plot_training_curves(eegnet, "EEGNet")
plot_training_curves(deepconvnet, "DeepConvNet")
plot_training_curves(shallowconvnet, "ShallowConvNet")

#Evaluate Models on Test Data
test_loss, test_acc = model1.evaluate(X_test, y_test)
print("EEGNet test accuracy:", test_acc)

test_loss, test_acc = model2.evaluate(X_test, y_test)
print("DeepConvNet test accuracy:", test_acc)

test_loss, test_acc = model3.evaluate(X_test, y_test)
print("ShallowConvNet test accuracy:", test_acc)

#Generate Classification Reports and Confusion Matrices for Models
y_pred1 = model1.predict(X_test).argmax(axis=1)
y_pred2 = model2.predict(X_test).argmax(axis=1)
y_pred3 = model3.predict(X_test).argmax(axis=1)
y_true = y_test.argmax(axis=1)

for y_pred, name in zip([y_pred1, y_pred2, y_pred3], ["EEGNet", "DeepConvNet", "ShallowConvNet"]):
    print(f"\n{name} Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print(f"{name} Confusion Matrix:\n{cm}")

#Compute F1 Scores, Model Parameters, and Inference Times
f1_1 = f1_score(y_true, y_pred1, average='macro')
f1_2 = f1_score(y_true, y_pred2, average='macro')
f1_3 = f1_score(y_true, y_pred3, average='macro')

inference_times = []
params_list = []
for model, name in zip([model1, model2, model3], ["EEGNet", "DeepConvNet", "ShallowConvNet"]):
    params = model.count_params()
    start = time.time()
    _ = model.predict(X_test[:32])
    inference_time = time.time() - start
    params_list.append(params)
    inference_times.append(inference_time)
    print(f"{name} params: {params}, inference time (32 samples): {inference_time:.4f}s")


results = pd.DataFrame({
    "Model": ["EEGNet", "DeepConvNet", "ShallowConvNet"],
    "Test Accuracy": [model1.evaluate(X_test, y_test, verbose=0)[1],
                      model2.evaluate(X_test, y_test, verbose=0)[1],
                      model3.evaluate(X_test, y_test, verbose=0)[1]],
    "F1-score (macro)": [f1_1, f1_2, f1_3],
    "Params": params_list,
    "Inference Time (32 samples)": inference_times
})

print("\n=== Model Comparison ===")
print(results)