
# 🧠 Spam Email Classifier Web App

This project is a machine learning-based web application that classifies email messages as **Spam** or **Ham (Not Spam)**. It uses a Naive Bayes classifier trained on a dataset of emails and provides a user-friendly interface via a Flask web app.

🌐 **Live Demo**: [https://spam-detection-app-ffsr.onrender.com/predict](https://spam-detection-app-ffsr.onrender.com/predict)

---

## 📂 Project Structure

```
.
├── train_model.py       # Script to clean data, train model, and save the pipeline
├── app.py               # Flask app for serving the web interface and predictions
├── spam_model.pkl       # ✅ Pre-trained model included for immediate use
├── templates/
│   └── index.html       # HTML frontend for user input and displaying prediction
├── requirements.txt     # Python dependencies
├── LICENSE              # MIT License
└── email.csv            # CSV dataset with email text and spam labels (not included here)
```

---
📊 Dataset
This project uses the Spam Email Dataset from Kaggle:
👉 [[https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset]
]([https://](https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset))
The dataset contains labeled messages as spam or ham.

Download the dataset and save it as email.csv in the root directory of this project.

The default columns expected are:

text: the message content

spam: the label (1 for spam, 0 for not spam)

## 🚀 Features

- Clean and preprocesses email text (removes links, HTML, punctuation, etc.)
- Uses `TfidfVectorizer` and `Multinomial Naive Bayes` for classification
- Flask-powered UI to input messages and view predictions
- Probability/confidence score displayed with each prediction
- ✅ Includes pre-trained model (`spam_model.pkl`) for instant deployment

---

## 🛠️ Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/spam-email-classifier.git
cd spam-email-classifier
```

### 2. Install Dependencies

Make sure you have Python 3.7+ installed. Then install required packages:

```bash
pip install -r requirements.txt
```

### 3. Train the Model (Optional)

You can use the included `spam_model.pkl` directly, or retrain with your own dataset:

```bash
python train_model.py
```

### 4. Run the Web App Locally

Start the Flask application:

```bash
python app.py
```

Then open your browser and go to `http://localhost:5000`.

---

## 🧪 Example

You enter:
```
Congratulations! You've won a $1,000 Walmart gift card. Click here to claim.
```

The app predicts:
```
Prediction: SPAM (Confidence: 98.25%)
```

---

## 🌐 Live App

Try the app live here:  
👉 **[https://spam-detection-app-ffsr.onrender.com/predict](https://spam-detection-app-ffsr.onrender.com/predict)**

---

## 📌 Notes

- The app loads the model (`spam_model.pkl`) at startup.
- If retraining, make sure to overwrite the old `spam_model.pkl`.
- Training uses 80/20 train-test split with stratification to preserve class balance.
- Prediction route handles errors gracefully and gives user-friendly messages.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙌 Acknowledgments

- Built with [Flask](https://flask.palletsprojects.com/)
- Machine Learning using [scikit-learn](https://scikit-learn.org/)

