# Titanic Survival Prediction App 🚢

This is a machine learning-powered interactive app that predicts a passenger's chances of survival on the Titanic based on various factors like age, gender, class, and more. The app integrates data visualization, predictive modeling, and an engaging user interface for an educational and exploratory experience.

## Features
1. Interactive Prediction
Users can manually input passenger information (age, sex, class, family size, etc.) to predict survival chances.
Provides a probability score and detailed feedback on the prediction.
2. Data Visualization
Decision Tree Visualization: Displays a graphical representation of a decision tree from the Random Forest model, explaining how predictions are made.
ROC Curve Visualization: Illustrates the performance of the predictive model using a Receiver Operating Characteristic curve and highlights the model's AUC (Area Under the Curve) score.
3. Model Training & Evaluation
Utilizes the Titanic dataset from Seaborn for training.
Includes data preprocessing steps like encoding categorical variables and standardizing features.
Implements a Random Forest Classifier, with cross-validation for performance assessment.
4. User-Friendly Interface
Built using the Tkinter library for a clean and simple graphical interface.
Intuitive design allows users to easily explore the model's capabilities.
Technical Stack
Machine Learning: Random Forest Classifier from sklearn.
Visualization: Matplotlib, Seaborn, and tree.plot_tree for decision tree display.
Frontend: Tkinter for GUI.
