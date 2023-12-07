from matplotlib import pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


class CarModel:
    """A class to handle various modeling processes for car data analysis."""

    def __init__(self, file_path):
        """Initialize the model with the dataset file path."""
        self.data = pd.read_csv(file_path)
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4

    def preprocess_and_split(self):
        """One-hot encode features and split the dataset into training and testing sets."""
        encoder = OneHotEncoder()
        X = encoder.fit_transform(self.data.drop("class", axis=1))
        y = self.data["class"].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    def build_and_evaluate_model(self):
        """Build a RandomForest model within a pipeline and evaluate it using cross-validation."""
        categorical_columns = self.data.drop("class", axis=1).columns
        preprocessor = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(), categorical_columns)]
        )

        model_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(random_state=42)),
            ]
        )

        cv_scores = cross_val_score(
            model_pipeline, self.data.drop("class", axis=1), self.data["class"], cv=5
        )
        print("Cross-validation scores:", cv_scores)
        print("Mean CV score:", cv_scores.mean())

    def tune_hyperparameters(self):
        """Tune hyperparameters of the RandomForest model using GridSearchCV."""
        categorical_columns = self.data.drop("class", axis=1).columns
        preprocessor = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(), categorical_columns)]
        )

        model_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(random_state=42)),
            ]
        )

        param_grid = {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [10, 20, 30],
            "classifier__min_samples_split": [2, 4, 6],
        }
        grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
        grid_search.fit(self.data.drop("class", axis=1), self.data["class"])

        best_parameters = grid_search.best_params_
        best_score = grid_search.best_score_

        print("Best Parameters:", best_parameters)
        print("Best CV Score:", best_score)

        self.best_parameters = grid_search.best_params_
        self.best_score = grid_search.best_score_

    def get_best_parameters(self):
        """Retrieve the best parameters found by hyperparameter tuning."""
        return self.best_parameters

    def train_and_evaluate_final_model(self, best_parameters):
        """Train and evaluate the final model using the best hyperparameters."""
        final_model = RandomForestClassifier(
            max_depth=best_parameters["classifier__max_depth"],
            min_samples_split=best_parameters["classifier__min_samples_split"],
            n_estimators=best_parameters["classifier__n_estimators"],
            random_state=42,
        )

        final_model.fit(self.X_train, self.y_train)
        final_predictions = final_model.predict(self.X_test)
        final_confusion_matrix = confusion_matrix(self.y_test, final_predictions)
        final_classification_report = classification_report(
            self.y_test, final_predictions
        )

        print("Final Model Confusion Matrix:\n", final_confusion_matrix)
        print("\nFinal Model Classification Report:\n", final_classification_report)

    def preprocess_and_split(self):
        """Encode features and split the dataset for 'maint' prediction."""
        X = self.data.drop("maint", axis=1)
        y = self.data["maint"]
        encoder = OneHotEncoder()
        X_encoded = encoder.fit_transform(X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_encoded, y, test_size=0.2, random_state=42
        )

    def build_model_and_evaluate(self):
        """Build and evaluate a RandomForest model using cross-validation."""
        model_pipeline = Pipeline(
            [("classifier", RandomForestClassifier(random_state=42))]
        )

        cv_scores = cross_val_score(model_pipeline, self.X_train, self.y_train, cv=5)
        print("Cross-validation scores:", cv_scores)
        print("Mean CV score:", cv_scores.mean())

    def tune_hyperparameterss(self):
        """Tune hyperparameters of the RandomForest model using GridSearchCV."""
        param_grid = {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [10, 20, 30],
            "classifier__min_samples_split": [2, 4, 6],
        }

        model_pipeline = Pipeline(
            [("classifier", RandomForestClassifier(random_state=42))]
        )

        grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)

        best_parameters = grid_search.best_params_
        best_score = grid_search.best_score_

        print("Best Parameters:", best_parameters)
        print("Best CV Score:", best_score)

        # Update the model with best parameters
        self.model_pipeline = Pipeline(
            [("classifier", RandomForestClassifier(**best_parameters, random_state=42))]
        )

    def evaluate_final_model(self):
        """Evaluate the final model on the test set."""
        # Ensure the model is updated with the best parameters
        final_model = RandomForestClassifier(**self.best_parameters, random_state=42)

        # Fit the model with training data
        final_model.fit(self.X_train, self.y_train)

        # Predict on the test set
        y_pred_test = final_model.predict(self.X_test)

        # Print classification report and confusion matrix
        print(
            "Classification Report (Test Set):\n",
            classification_report(self.y_test, y_pred_test),
        )
        print(
            "Confusion Matrix (Test Set):\n", confusion_matrix(self.y_test, y_pred_test)
        )

    def one_hot_encode_dataset(self):
        """One-hot encode the entire dataset."""
        encoder = OneHotEncoder()
        X_encoded = encoder.fit_transform(self.data)
        X_encoded_df = pd.DataFrame(
            X_encoded.toarray(),
            columns=encoder.get_feature_names_out(self.data.columns),
        )
        return X_encoded_df

    def apply_kmeans_and_elbow_method(self, X_encoded_df):
        """Apply KMeans clustering and use the Elbow Method to determine optimal clusters."""
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
            kmeans.fit(X_encoded_df)
            wcss.append(kmeans.inertia_)

        plt.plot(range(1, 11), wcss)
        plt.title("Elbow Method")
        plt.xlabel("Number of clusters")
        plt.ylabel("WCSS")
        plt.show()

    def apply_kmeans_and_analyze_clusters(self, X_encoded_df, num_clusters):
        """Apply KMeans clustering and analyze the resulting clusters."""
        kmeans = KMeans(n_clusters=num_clusters, init="k-means++", random_state=42)
        clusters = kmeans.fit_predict(X_encoded_df)
        self.data["Cluster"] = clusters

        for cluster in range(num_clusters):
            cluster_data = self.data[self.data["Cluster"] == cluster]
            print(f"Analysis of Cluster {cluster}:")
            print(cluster_data.describe(include="all"), "\n")
