import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from imblearn.over_sampling import SMOTE  # Import SMOTE for oversampling

# load the data
X_scaled = pd.read_csv("data/processed/X_scaled.csv")  # Combined audio + text features
y = pd.read_csv("data/processed/y.csv").values.ravel()  # Targets (0: Control, 1: Dementia)
y = np.where(y == 'Dementia', 1, 0).astype(float)

# split into audio and text features
audio_feature_dim = 50  # Adjust based on the number of audio features
audio_features = X_scaled.iloc[:, :audio_feature_dim].values  
text_features = X_scaled.iloc[:, audio_feature_dim:].values 

# stratified train-test split
X_audio_train, X_audio_test, X_text_train, X_text_test, y_train, y_test = train_test_split(
    audio_features, text_features, y, test_size=0.2, stratify=y, random_state=42
)

# check class distribution in training and test sets
print("Training set class distribution:", np.unique(y_train, return_counts=True))
print("Test set class distribution:", np.unique(y_test, return_counts=True))

# compute class weights
if len(np.unique(y_train)) == 2:
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = {0: class_weights[0], 1: class_weights[1]}
    print("Class Weights:", class_weights_dict)
else:
    print("Warning: Only one class present in training data. Adjust the dataset or split.")
    class_weights_dict = None

# multi-branch neural network architecture
# audio input branch
audio_input = Input(shape=(X_audio_train.shape[1],), name="audio_input")
audio_dense1 = Dense(128, activation="relu", kernel_regularizer=l2(0.01))(audio_input)
audio_dense1 = BatchNormalization()(audio_dense1)
audio_dense2 = Dense(64, activation="relu", kernel_regularizer=l2(0.01))(audio_dense1)

# text input branch
text_input = Input(shape=(X_text_train.shape[1],), name="text_input")
text_dense1 = Dense(128, activation="relu", kernel_regularizer=l2(0.01))(text_input)
text_dense1 = BatchNormalization()(text_dense1)
text_dense2 = Dense(64, activation="relu", kernel_regularizer=l2(0.01))(text_dense1)

# fusion layer
fusion = Concatenate()([audio_dense2, text_dense2])  # Combine audio and text features
fusion_dense = Dense(128, activation="relu", kernel_regularizer=l2(0.01))(fusion)
dropout = Dropout(0.6)(fusion_dense)  # Increased dropout
output = Dense(1, activation="sigmoid")(dropout)  # Sigmoid for binary classification

# build the model
model = Model(inputs=[audio_input, text_input], outputs=output)

# compile with a lower learning rate
model.compile(optimizer=Adam(learning_rate=0.0001), loss="binary_crossentropy", metrics=["accuracy"])

# display model summary
model.summary()

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# train the model
history = model.fit(
    [X_audio_train, X_text_train], y_train,
    validation_split=0.2,  
    epochs=50,  
    batch_size=32,
    verbose=1,
    class_weight=class_weights_dict, 
    callbacks=[early_stopping]
)

# evaluate the model
test_loss, test_accuracy = model.evaluate([X_audio_test, X_text_test], y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")

# generate predictions
y_pred_prob = model.predict([X_audio_test, X_text_test])
y_pred = (y_pred_prob > 0.5).astype("int32")

print("Unique predictions in test set:", np.unique(y_pred, return_counts=True))

# classification report and metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))

precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)
print(f"Per-class metrics:")
print(f"Precision (class 0): {precision[0]:.4f}, Recall (class 0): {recall[0]:.4f}, F1 (class 0): {f1[0]:.4f}")
print(f"Precision (class 1): {precision[1]:.4f}, Recall (class 1): {recall[1]:.4f}, F1 (class 1): {f1[1]:.4f}")

roc_auc = roc_auc_score(y_test, y_pred_prob)
print(f"ROC-AUC Score: {roc_auc:.4f}")

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Control', 'Dementia'], yticklabels=['Control', 'Dementia'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

model.save("models/dementia_multi_modal_model.h5")
