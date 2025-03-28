import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score

# Load data
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Encode labels
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels)
onehot_encoder = OneHotEncoder(sparse_output=False)
labels = onehot_encoder.fit_transform(integer_encoded.reshape(-1, 1))

# Reshape data for LSTM (samples, timesteps, features)
data = data.reshape(data.shape[0], 1, data.shape[1])

# Split data
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Build LSTM model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(1, data.shape[2])),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(labels.shape[1], activation='softmax')  # Output layer
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=50, batch_size=16, validation_data=(x_test, y_test))

# Evaluate the model
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

score = accuracy_score(y_test_classes, y_pred_classes)
print('{}% of samples were classified correctly!'.format(score * 100))

# Save the trained model
with open('model1.p', 'wb') as f:
    pickle.dump({'model': model, 'label_encoder': label_encoder}, f)
