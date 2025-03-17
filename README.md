# Water Quality Classification Project

## Overview
This project implements a neural network model to classify water quality data based on various parameters. The model predicts the site ID (location) using water quality measurements, potentially helping to identify the source of water samples based on their characteristics.

## Features
- Multi-class classification of water quality data
- Neural network implementation using TensorFlow/Keras
- Data preprocessing including scaling and encoding
- Model evaluation with confusion matrix visualization

## Dataset
The project uses a dataset (`clean_data_water_quality_fin.xlsx`) containing water quality measurements from different sites. The data includes various water quality parameters with the target variable being the site identifier.

## Technical Implementation
- **Neural Network Architecture**: A deep neural network with multiple dense layers and dropout regularization
- **Preprocessing**: StandardScaler for feature scaling and OneHotEncoder for categorical target encoding
- **Model Configuration**:
  - 3 hidden layers (256, 128, 64 neurons)
  - ReLU activation functions
  - Dropout rate of 0.35 for regularization
  - Softmax activation for output layer
  - Adam optimizer with learning rate of 0.0015
  - Categorical crossentropy loss function

## Requirements
- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- torch
- scikit-learn
- TensorFlow/Keras

## Usage
1. Install the required libraries:
   ```
   pip install pandas numpy matplotlib seaborn torch scikit-learn tensorflow
   ```

2. Load the dataset:
   ```python
   df = pd.read_excel("clean_data_water_quality_fin.xlsx")
   ```

3. Preprocess the data:
   ```python
   X = df.drop(columns=["Site_Id"])
   y = df["Site_Id"]
   label_encoder = LabelEncoder()
   y_encoded = label_encoder.fit_transform(y)
   y_onehot = to_categorical(y_encoded)
   ```

4. Train the model:
   ```python
   nnModel.fit(X_train_scaled, y_train, epochs=100, batch_size=16, validation_split=0.2)
   ```

5. Evaluate the model:
   ```python
   loss, accuracy = nnModel.evaluate(X_test_scaled, y_test)
   ```

## Results
The model achieves approx. 60% accuracy in classifying water samples to their originating sites. The confusion matrix visualization helps in understanding the model's performance across different classes.
