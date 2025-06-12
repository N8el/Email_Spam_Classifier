# Email Spam Classifier

## Project Overview

This project implements an **Email Spam Classifier** using machine learning techniques to accurately distinguish between legitimate ("ham") and unsolicited ("spam") emails. Email spam remains a significant issue, impacting user productivity and potentially posing security risks. By building an effective classifier, this project aims to provide a robust solution for filtering unwanted messages and improving the overall email experience.

## Features

* **Data Loading and Preprocessing:** Handles raw email text data, including label encoding (`spam` to `0`, `ham` to `1`).
* **TF-IDF Feature Extraction:** Converts textual email content into numerical feature vectors suitable for machine learning algorithms.
* **Logistic Regression Model:** Utilizes a Logistic Regression model for efficient and accurate binary classification.
* **Model Training & Evaluation:** Splits data into training and testing sets to assess model performance using accuracy metrics.
* **Interactive Prediction:** Allows users to input new email content and receive a real-time "ham" or "spam" prediction.

## Technologies Used

* **Python 3.x**
* **Jupyter Notebook:** For interactive development and presentation.
* **NumPy:** For numerical operations.
* **Pandas:** For data manipulation and analysis.
* **Scikit-learn (sklearn):** For machine learning functionalities, including:
    * `train_test_split` for dataset division.
    * `TfidfVectorizer` for feature extraction.
    * `LogisticRegression` for the classification model.
    * `accuracy_score` for model evaluation.

## Dataset

The model was trained on a dataset containing a collection of emails labeled as either 'spam' or 'ham'. The dataset is expected to be in a CSV format, typically named `mail_data.csv`, with at least two columns: one for the email `Message` and another for its `Category` (ham/spam).

## How to Run the Project

To run this project on your local machine:

1.  **Clone the repository (or download the files):**
    If you've cloned it:
    ```bash
    git clone [https://github.com/YourGitHubUsername/Email-Spam-Classifier.git](https://github.com/YourGitHubUsername/Email-Spam-Classifier.git)
    cd Email-Spam-Classifier
    ```
    (Replace `YourGitHubUsername` and `Email-Spam-Classifier` with your actual details)

2.  **Ensure you have the necessary libraries installed:**
    It's recommended to use a virtual environment.
    ```bash
    pip install numpy pandas scikit-learn jupyter
    ```

3.  **Place the dataset:** Make sure the `mail_data.csv` file is in the same directory as your Jupyter Notebook.

4.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```

5.  **Open the notebook:** In the Jupyter interface that opens in your browser, click on your project's `.ipynb` file (e.g., `Email_Spam_Classifier.ipynb`).

6.  **Run all cells:** Execute all cells in the notebook sequentially to see the data loading, preprocessing, model training, evaluation, and the interactive prediction example.

## Model Performance

The trained Logistic Regression model achieved the following approximate accuracies:

* **Accuracy on Training Data:** ~97.7%
* **Accuracy on Test Data:** ~96.8%

These results indicate a strong performance in classifying both seen and unseen email data.

## Future Enhancements

* **Exploring Other Models:** Investigate the performance of other machine learning algorithms (e.g., Naive Bayes, SVMs, Random Forest) or even deep learning models (e.g., LSTMs, Transformers).
* **Advanced Preprocessing:** Implement more sophisticated text normalization techniques such as stemming, lemmatization, or handling of special characters and URLs.
* **Model Deployment:** Integrate the trained model into a web application or an email client for real-time spam detection.
* **Larger and More Diverse Datasets:** Train the model on a broader and more varied collection of emails to enhance its generalization capabilities.

---
