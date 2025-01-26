# Crop Recommendation System

## Overview
The Crop Recommendation System is a machine learning-based project that assists farmers in selecting the most suitable crop to grow based on specific environmental conditions. By analyzing key parameters like soil nutrients, temperature, humidity, pH, and rainfall, this system predicts the optimal crop to maximize yield.

---

## Dataset
The dataset used for this project contains 2200 records with the following columns:

- **N**: Nitrogen content in the soil
- **P**: Phosphorus content in the soil
- **K**: Potassium content in the soil
- **temperature**: Ambient temperature (in degrees Celsius)
- **humidity**: Relative humidity (in percentage)
- **ph**: Soil pH value
- **rainfall**: Rainfall (in mm)
- **label**: Target column representing crop type (e.g., rice, maize, etc.)

The dataset has no missing or duplicate values, ensuring clean and ready-to-use data.

---

## Features and Implementation
### **1. Data Exploration**
- Basic statistics (mean, median, etc.) using `df.describe()`.
- Checked for null and duplicate values.
- Visualized correlations between features using a heatmap (`sns.heatmap`).

### **2. Train-Test Split**
Split the dataset into training and testing sets:
- **Features (X)**: ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
- **Target (y)**: ['label']

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

---

## Models Implemented
The following machine learning models were trained and tested:

### **1. Logistic Regression**
Achieved an accuracy of **94.85%**.

### **2. Decision Tree**
- Criterion: Entropy
- Maximum depth: 5
- Accuracy: **86.97%**

### **3. Gaussian Naive Bayes**
Achieved the highest accuracy of **99.39%**.

### **4. Support Vector Machine (SVM)**
- Kernel: Polynomial
- Degree: 3
- Normalized the data before training.
- Accuracy: **96.82%**

### **5. Random Forest**
- Number of estimators: 20
- Accuracy: **99.24%**.

---

## Model Comparison
The accuracies of the models are compared using a bar plot:

```python
sns.barplot(x=acc, y=model)
plt.title('Accuracy Comparison')
plt.xlabel('Accuracy')
plt.ylabel('Algorithm')
```

### **Accuracy Results**
| Model                | Accuracy   |
|----------------------|------------|
| Logistic Regression  | 94.85%     |
| Decision Tree        | 86.97%     |
| Gaussian Naive Bayes | 99.39%     |
| SVM                  | 96.82%     |
| Random Forest        | 99.24%     |

---

## Prediction Example
The trained Random Forest model was used for prediction. An example input and result:

**Input**:
```python
[104, 18, 30, 23.603016, 60.3, 6.7, 140.91]
```
**Output**:
```python
['coffee']
```

---

## Dependencies
Install the required libraries using the `requirements.txt` file:

```
numpy==1.24.3
pickle-mixin==1.0.2
streamlit==1.23.1
seaborn==0.10.1
pandas==2.0.2
matplotlib==3.7.1
scikit_learn==1.2.2
```

To install dependencies:
```bash
pip install -r requirements.txt
```

---

## Running the Project
1. Clone the repository.
2. Install the dependencies using the `requirements.txt` file.
3. Run the Python script to train and test the model.
4. Optionally, deploy the model with a web UI using Streamlit.

---

## Future Enhancements
- Improve model accuracy with hyperparameter tuning.
- Integrate additional features like soil type and weather patterns.
- Develop a mobile-friendly user interface for ease of access.

---

