# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn import tree
import tkinter as tk
from tkinter import messagebox

# Load Titanic dataset from seaborn library
data = sns.load_dataset("titanic")

# Basic data cleaning
# Drop columns that are not useful for the prediction model
data = data.drop(["deck", "embark_town", "alive"], axis=1)
# Fill missing 'age' with median value, as it's a numeric column
data["age"].fillna(data["age"].median())
# Fill missing 'embarked' with mode (most frequent value), as it's categorical
data["embarked"].fillna(data["embarked"].mode()[0])

# Encode categorical variables using LabelEncoder
# LabelEncoder converts categories into numeric labels for model processing
label_encoders = {}
for col in ["sex", "embarked", "class", "who", "adult_male"]:
    le = LabelEncoder()  # Initialize LabelEncoder for each column
    data[col] = le.fit_transform(data[col])  # Encode the column
    label_encoders[col] = le  # Store the encoder for potential inverse transformation

# Define feature variables (X) and target variable (y)
X = data.drop(["survived"], axis=1)  # Features without the target column
y = data["survived"]  # Target variable is 'survived'

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Select the features that will be used for training
features_to_use = ["age", "sex", "pclass", "sibsp", "parch", "embarked"]
X_train_selected = X_train[features_to_use]  # Training features
X_test_selected = X_test[features_to_use]  # Test features

# Standardize features (mean=0, variance=1) for better performance in some models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(
    X_train_selected
)  # Fit and transform the training set
X_test_scaled = scaler.transform(
    X_test_selected
)  # Transform the test set using the same scaler

# Initialize Random Forest Classifier with hyperparameters and train the model
rf = RandomForestClassifier(
    n_estimators=200, max_depth=10, random_state=42, class_weight="balanced"
)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)  # Predict survival for the test set

# Evaluate the model with cross-validation to assess performance
cross_val_score_rf = cross_val_score(
    rf, X_train_scaled, y_train, cv=5, scoring="accuracy"
)
print(f"Cross-Validation Scores: {cross_val_score_rf}")
print(f"Mean Cross-Validation Accuracy: {cross_val_score_rf.mean():.4f}")

# Create a tkinter window for user interface
window = tk.Tk()
window.title("Titanic Survival Prediction")
window.geometry("600x500")  # Set the size of the window

# Define font style to be used in all widgets
font_style = ("Helvetica", 12)  # Font type and size


# Function to show decision tree visualization (from Random Forest)
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Global variables to store references to the windows
decision_tree_window = None
roc_curve_window = None
feature_importance_window = None

# Function to show decision tree visualization (from Random Forest)
def show_decision_tree():
    global decision_tree_window
    if decision_tree_window is not None and decision_tree_window.winfo_exists():
        decision_tree_window.destroy()  # Close the existing decision tree window

    # Create a new top-level window
    decision_tree_window = tk.Toplevel(window)
    decision_tree_window.title("Decision Tree Visualization")
    decision_tree_window.geometry("800x600")  # Adjust the size

    intro_text = """
    The decision tree is a model that splits the data into subsets based on the most significant features to predict the outcome (survival).
    Each decision point in the tree represents a test on a feature, and the branches represent possible outcomes. The leaf nodes represent the final prediction.
    This visualization helps us understand how the model makes its decisions and which features play a key role in predicting survival.
    """

    # Add explanation text in the new window
    tk.Label(decision_tree_window, text="Decision Tree", font=("Helvetica", 14, "bold")).pack(pady=10)
    tk.Label(decision_tree_window, text=intro_text, font=font_style, justify=tk.LEFT, wraplength=650).pack(pady=10)

    # Plot the first decision tree from the Random Forest model
    fig, ax = plt.subplots(figsize=(12, 8))  # Create a Matplotlib figure
    tree.plot_tree(
        rf.estimators_[0],  # Use the first tree from the random forest
        feature_names=features_to_use,  # Feature names for better readability
        class_names=["Not Survived", "Survived"],  # Class names (target variable)
        filled=True,  # Color nodes based on majority class
        ax=ax,  # Attach the plot to the axes
    )
    plt.title("Random Forest Decision Tree Visualization")

    # Embed the plot in the Tkinter window using FigureCanvasTkAgg
    canvas = FigureCanvasTkAgg(fig, master=decision_tree_window)  # Create a canvas for the figure
    canvas.draw()  # Draw the plot on the canvas
    canvas.get_tk_widget().pack(pady=20)

# Function to show ROC curve (Receiver Operating Characteristic)
def show_roc_curve():
    global roc_curve_window
    if roc_curve_window is not None and roc_curve_window.winfo_exists():
        roc_curve_window.destroy()  # Close the existing ROC curve window

    # Create a new top-level window
    roc_curve_window = tk.Toplevel(window)
    roc_curve_window.title("ROC Curve Visualization")
    roc_curve_window.geometry("800x600")  # Adjust the size

    intro_text = """
    An ROC curve (Receiver Operating Characteristic curve) is a graphical representation of a binary classifier's performance across different threshold values. 
    It illustrates how well the model distinguishes between the two classes (e.g., positive vs. negative, survived vs. not survived). 
    The ROC curve is used to evaluate the trade-offs between sensitivity (true positive rate) and specificity (false positive rate) at various thresholds.
    """

    # Add explanation text in the new window
    tk.Label(roc_curve_window, text="ROC Curve", font=("Helvetica", 14, "bold")).pack(pady=10)
    tk.Label(roc_curve_window, text=intro_text, font=font_style, justify=tk.LEFT, wraplength=650).pack(pady=10)

    # Get the predicted probabilities for the test set
    y_pred_prob_rf = rf.predict_proba(X_test_scaled)[:, 1]
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_prob_rf)  # Calculate fpr and tpr
    roc_auc_rf = auc(fpr_rf, tpr_rf)  # Calculate AUC

    # Plot the ROC curve
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {roc_auc_rf:.2f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random Guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()

    # Embed the plot in the Tkinter window using FigureCanvasTkAgg
    canvas = FigureCanvasTkAgg(fig, master=roc_curve_window)
    canvas.draw()
    canvas.get_tk_widget().pack(pady=20)

# Function for manual input and survival prediction based on user input
def predict_survival():
    input_window = tk.Toplevel(window)
    input_window.title("Enter Passenger Information")
    input_window.geometry("600x600")

    # Titanic scenario description text for the user
    intro_text = """
The year is 1912. You are aboard the R.M.S. Titanic, the largest and most luxurious ship in the world. It's your maiden voyage, but disaster strikes when the Titanic hits an iceberg in the cold waters of the North Atlantic.

In the chaos, lifeboats are limited, and survival depends on various factors such as your age, gender, class, and whether you were able to board a lifeboat in time. The officers are selecting passengers for the lifeboats based on priority, and your fate is uncertain.

You boarded the Titanic at one of these ports:
C (Cherbourg): A French port for wealthier passengers (higher survival rates).
Q (Queenstown): A smaller Irish port with mixed survival rates.
S (Southampton): The main English port, with generally lower survival rates.

Will you survive this tragic event? Let's see if we can predict your chances of survival based on the information you provide.
    """
    # Display the Titanic scenario text in the new window
    tk.Label(
        input_window,
        text="Titanic Scenario",
        font=("Helvetica", 14, "bold"),
        justify=tk.LEFT,
        wraplength=650,
    ).pack(pady=10)
    tk.Label(
        input_window, text=intro_text, font=font_style, justify=tk.LEFT, wraplength=480
    ).pack(pady=10)

    # Function to handle form submission
    def submit_form():
        try:
            # Collect user inputs from the entry fields and validate them
            age = age_entry.get().strip()
            if age == "" or not age.replace(".", "", 1).isdigit():
                raise ValueError("Age must be a valid number.")
            age = float(age)

            sex = sex_entry.get().strip()
            if sex not in ["0", "1"]:
                raise ValueError("Sex must be 0 (male) or 1 (female).")
            sex = int(sex)

            pclass = class_entry.get().strip()
            if pclass not in ["1", "2", "3"]:
                raise ValueError("Pclass must be 1, 2, or 3.")
            pclass = int(pclass)

            sibsp = sibsp_entry.get().strip()
            parch = parch_entry.get().strip()
            if not sibsp.isdigit() or not parch.isdigit():
                raise ValueError("SibSp and Parch must be valid integers.")
            sibsp = int(sibsp)
            parch = int(parch)

            embarked = embarked_entry.get().strip()
            if embarked == "":  # Default value for Embarked if not entered
                embarked = 0
            elif embarked not in ["0", "1", "2"]:
                raise ValueError("Embarked must be 0 (C), 1 (Q), or 2 (S).")
            else:
                embarked = int(embarked)

            # Create input array based on the form data
            input_data = [[age, sex, pclass, sibsp, parch, embarked]]
            input_scaled = scaler.transform(input_data)  # Scale the input features

            # Predict survival using the Random Forest model
            prediction = rf.predict(input_scaled)
            survival_probability = rf.predict_proba(input_scaled)[0][
                1
            ]  # Probability of survival

            # Display the result in a message box
            survival_reason = f"Based on the details provided, you {'survived' if prediction == 1 else 'did not survive'}.\n"
            survival_reason += f"Your predicted chance of survival is {survival_probability * 100:.2f}%. "
            survival_reason += f"Confidence: {round(survival_probability * 100, 1)}%"
            messagebox.showinfo("Prediction Result", survival_reason)  # Show result
            input_window.destroy()  # Close the input window after submitting

        except ValueError as e:
            messagebox.showerror(
                "Input Error", str(e)
            )  # Show error message if input is invalid

    # Create form fields to collect user data (age, sex, etc.)
    tk.Label(input_window, text="Age:", font=font_style).pack(pady=5)
    age_entry = tk.Entry(input_window, font=font_style)
    age_entry.pack(pady=5)

    tk.Label(input_window, text="Sex (0 = male, 1 = female):", font=font_style).pack(
        pady=5
    )
    sex_entry = tk.Entry(input_window, font=font_style)
    sex_entry.pack(pady=5)

    tk.Label(input_window, text="Pclass (1, 2, 3):", font=font_style).pack(pady=5)
    class_entry = tk.Entry(input_window, font=font_style)
    class_entry.pack(pady=5)

    tk.Label(input_window, text="Siblings/Spouse (SibSp):", font=font_style).pack(
        pady=5
    )
    sibsp_entry = tk.Entry(input_window, font=font_style)
    sibsp_entry.pack(pady=5)

    tk.Label(input_window, text="Parents/Children (Parch):", font=font_style).pack(
        pady=5
    )
    parch_entry = tk.Entry(input_window, font=font_style)
    parch_entry.pack(pady=5)

    tk.Label(
        input_window, text="Embarked (0 = C, 1 = Q, 2 = S):", font=font_style
    ).pack(pady=5)
    embarked_entry = tk.Entry(input_window, font=font_style)
    embarked_entry.pack(pady=5)

    submit_btn = tk.Button(
        input_window, text="Submit", font=font_style, command=submit_form
    )
    submit_btn.pack(pady=15)

    # Run the input window (starts tkinter event loop for this window)
    input_window.mainloop()


# Create the main buttons with bigger fonts
btn_decision_tree = tk.Button(
    window, text="Show Decision Tree", font=font_style, command=show_decision_tree
)
btn_decision_tree.pack(pady=10)

btn_roc_curve = tk.Button(window, text="Show ROC Curve", font=font_style, command=show_roc_curve)
btn_roc_curve.pack(pady=10)

btn_manual_input = tk.Button(window, text="Predict Survival", font=font_style, command=predict_survival)
btn_manual_input.pack(pady=10)

# Run the main tkinter window (event loop for the app)
window.mainloop()
