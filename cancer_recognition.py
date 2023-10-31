# Import necessary libraries
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def load_data(file_path):
    """
    Load the dataset from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    dataset = pd.read_csv(file_path)
    return dataset

def preprocess_data(dataset):
    """
    Preprocess the dataset by separating features and labels.

    Parameters:
        dataset (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame, pd.Series: Features (x) and labels (y).
    """
    x = dataset.drop(columns=['diagnosis(1=m, 0=b)'])
    y = dataset['diagnosis(1=m, 0=b)']
    return x, y

def split_data(x, y, test_size=0.2):
    """
    Split the data into training and testing sets.

    Parameters:
        x (pd.DataFrame): Features.
        y (pd.Series): Labels.
        test_size (float, optional): Fraction of the data to reserve for testing. Default is 0.2.

    Returns:
        pd.DataFrame, pd.DataFrame, pd.Series, pd.Series: x_train, x_test, y_train, y_test
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    return x_train, x_test, y_train, y_test

def build_model(input_shape):
    """
    Build the neural network model.

    Parameters:
        input_shape (tuple): Shape of a single input sample.

    Returns:
        tf.keras.models.Sequential: Compiled neural network model.
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(256, input_shape=input_shape, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(256, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, x_train, y_train, epochs=1000):
    """
    Train the neural network model.

    Parameters:
        model (tf.keras.models.Sequential): Compiled neural network model.
        x_train (pd.DataFrame): Features for training.
        y_train (pd.Series): Labels for training.
        epochs (int, optional): Number of training epochs. Default is 1000.
    """
    model.fit(x_train, y_train, epochs=epochs)

def evaluate_model(model, x_test, y_test):
    """
    Evaluate the neural network model on the test set.

    Parameters:
        model (tf.keras.models.Sequential): Compiled neural network model.
        x_test (pd.DataFrame): Features for testing.
        y_test (pd.Series): Labels for testing.

    Returns:
        list: Test loss and accuracy.
    """
    return model.evaluate(x_test, y_test)

def calculate_metrics(model, x_test, y_test):
    """
    Calculate various evaluation metrics.

    Parameters:
        model (tf.keras.models.Sequential): Trained neural network model.
        x_test (pd.DataFrame): Features for testing.
        y_test (pd.Series): True labels for testing.

    Returns:
        dict: Dictionary containing precision, recall, true positives (tp), true negatives (tn),
              false positives (fp), false negatives (fn), and f1-score.
    """
    y_pred = model.predict(x_test)
    y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    metrics = {
        'Precision': precision,
        'Recall': recall,
        'True Positives': tp,
        'True Negatives': tn,
        'False Positives': fp,
        'False Negatives': fn,
        'F1-Score': f1,
        'Confusion Matrix': confusion_matrix(y_test, y_pred).tolist()
    }

    return metrics

if __name__ == "__main__":
    # Example usage
    dataset = load_data('cancer.csv')
    x, y = preprocess_data(dataset)
    x_train, x_test, y_train, y_test = split_data(x, y)
    model = build_model(input_shape=(x_train.shape[1],))
    train_model(model, x_train, y_train)
    test_results = evaluate_model(model, x_test, y_test)
    print("Model Summary:", test_results[0])
    print("Test Loss:", test_results[0])
    print("Test Accuracy:", test_results[1])
    metrics = calculate_metrics(model, x_test, y_test)
    print("Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")