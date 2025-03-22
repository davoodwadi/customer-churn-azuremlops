import argparse
from pathlib import Path
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser("train")
    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--test_data", type=str, help="Path to test dataset")
    parser.add_argument("--model_output", type=str, help="Path of output model")
    parser.add_argument('--criterion', type=str, default='gini', help='The function to measure the quality of a split')
    parser.add_argument('--max_depth', type=int, default=None, help='The maximum depth of the tree')
    return parser.parse_args()

def main(args):
    '''Read train and test datasets, train model, evaluate model, save trained model'''
    train_df = pd.read_csv(Path(args.train_data)/"train.csv")
    test_df = pd.read_csv(Path(args.test_data)/"test.csv")
    
    target_col = 'Churn'  # Assuming 'Churn' is the target column
    
    y_train = train_df[target_col]
    X_train = train_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    
    model = DecisionTreeClassifier(criterion=args.criterion, max_depth=args.max_depth)
    model.fit(X_train, y_train)
    
    mlflow.log_param("model", "DecisionTreeClassifier")
    mlflow.log_param("criterion", args.criterion)
    mlflow.log_param("max_depth", args.max_depth)
    
    yhat_test = model.predict(X_test)
    accuracy = accuracy_score(y_test, yhat_test)
    print(f'Accuracy of Decision Tree classifier on test set: {accuracy:.2f}')
    mlflow.log_metric("Accuracy", float(accuracy))
    
    mlflow.sklearn.save_model(sk_model=model, path=args.model_output)

if __name__ == "__main__":
    mlflow.start_run()
    args = parse_args()
    main(args)
    mlflow.end_run()
