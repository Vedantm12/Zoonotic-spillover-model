import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime

def train_disease_classifier(file_path, target_col, feature_cols=None):
    """
    Trains a Logistic Regression model and saves the performance report to a text file.
    """
    
    # 1. Load the dataset
    try:
        df = pd.read_csv(file_path)
        print(f"\nDataset '{file_path}' loaded successfully!")
    except FileNotFoundError:
        print(f"\nError: Could not find '{file_path}'. Make sure it is in the same folder.")
        return None

    # Drop rows with missing values
    df = df.dropna()

    # 2. Separate Features (X) and Target (y)
    try:
        if feature_cols:
            X = df[feature_cols]
        else:
            X = df.drop(columns=[target_col])
        y = df[target_col]
    except KeyError as e:
        print(f"Error: Column {e} not found in the dataset. Please check your spelling.")
        return None

    # 3. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 4. Initialize the Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000))
    ])

    # 5. Train and Predict
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # 6. Evaluation & Saving to File
    accuracy = accuracy_score(y_test, y_pred) * 100
    report = classification_report(y_test, y_pred, zero_division=0)
    
    # Format the output string
    output_string = (
        f"--- Model Performance for predicting '{target_col}' ---\n"
        f"Dataset: {file_path}\n"
        f"Accuracy: {accuracy:.2f}%\n\n"
        f"Classification Report:\n{report}\n"
    )
    
    # Print to the terminal
    print(f"\n{output_string}")
    
    # Create a unique filename using the target column and current time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{target_col}_results_{timestamp}.txt"
    
    # Write the string to a text file
    with open(output_filename, 'w') as f:
        f.write(output_string)
        
    print(f"✅ Results successfully saved to: {output_filename}\n")
    
    return pipeline

# TERMINAL INPUT LOGIC

if __name__ == "__main__":
    print("\n=== Generalized Disease Classifier ===")
    
    user_file = input("Enter the CSV file name : ").strip()
    user_target = input("Enter the target column to predict: ").strip()
    
    print("\nEnter the feature columns separated by commas.")
    print("Or just press ENTER to automatically use all other columns.")
    user_features_raw = input("Features: ").strip()
    
    if user_features_raw == "":
        user_features = None  
    else:
        user_features = [col.strip() for col in user_features_raw.split(",")]
        
    train_disease_classifier(file_path=user_file, target_col=user_target, feature_cols=user_features)
