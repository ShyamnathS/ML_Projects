# Sentiment Analysis using LSTM

## 📌 Project Overview
This project implements **Sentiment Analysis** on movie reviews using an **LSTM (Long Short-Term Memory) model**. The dataset used is the **IMDB Dataset of 50K Movie Reviews**, which consists of labeled reviews as either **positive** or **negative**. The model is trained using TensorFlow and Keras.

---

## 🔧 Tech Stack
- **Python**
- **TensorFlow / Keras**
- **Scikit-learn**
- **Pandas**
- **Google Colab** (for training the model)
- **GitHub** (for version control)

---

## 📂 Dataset
- The dataset is downloaded from Kaggle: [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- The dataset consists of two columns:
  - `review` (text of the review)
  - `sentiment` (either `positive` or `negative`)

---

## 🚀 Project Workflow

### 1️⃣ **Import Dependencies**
```python
import os
import json
import pandas as pd
from zipfile import ZipFile
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
```

### 2️⃣ **Download Dataset from Kaggle**
```python
!kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

# Unzip the dataset
with ZipFile("imdb-dataset-of-50k-movie-reviews.zip", "r") as zip_ref:
    zip_ref.extractall()
```

### 3️⃣ **Load and Preprocess Data**
```python
data = pd.read_csv("IMDB Dataset.csv")
data.replace({"sentiment": {"positive": 1, "negative": 0}}, inplace=True)

# Split dataset
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
```

### 4️⃣ **Tokenization and Padding**
```python
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(train_data["review"])

X_train = pad_sequences(tokenizer.texts_to_sequences(train_data["review"]), maxlen=200)
X_test = pad_sequences(tokenizer.texts_to_sequences(test_data["review"]), maxlen=200)

Y_train = train_data["sentiment"]
Y_test = test_data["sentiment"]
```

### 5️⃣ **Build LSTM Model**
```python
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=200))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
```

### 6️⃣ **Train the Model**
```python
model.fit(X_train, Y_train, epochs=5, batch_size=64, validation_split=0.2)
```

### 7️⃣ **Evaluate the Model**
```python
loss, accuracy = model.evaluate(X_test, Y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
```

### 8️⃣ **Predict Sentiment for a New Review**
```python
def predict_sentiment(review):
    sequence = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequence, maxlen=200)
    prediction = model.predict(padded_sequence)
    return "positive" if prediction[0][0] > 0.5 else "negative"

# Example usage
new_review = "This movie was fantastic. I loved it."
print(f"Sentiment: {predict_sentiment(new_review)}")
```

---

## 📊 Results
- The model achieves around **85-88% accuracy** on the test dataset.
- It can classify movie reviews as **positive** or **negative** based on the text input.

---

## 🛠 How to Run the Project
1. **Clone the Repository:**
   ```sh
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```
2. **Install Dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Run the Jupyter Notebook or Python Script.**

---

## 📌 Future Improvements
- Use **Bidirectional LSTMs** for better context understanding.
- Train on **larger datasets** for improved accuracy.
- Implement **attention mechanisms** to focus on important words.
- Deploy the model as a **web app using Flask or FastAPI**.

---

## 📜 License
This project is open-source and available under the **MIT License**.

---

## 💡 Author
Developed by **[Shyamnath S]**

If you liked this project, give it a ⭐ on GitHub!
