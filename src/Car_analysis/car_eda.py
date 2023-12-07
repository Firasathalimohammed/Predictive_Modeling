import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier


class CarDataEDA:
    """Class for performing exploratory data analysis on car data."""

    def __init__(self, file_path):
        """Initialize the class with the path to the dataset."""
        self.data = pd.read_csv(file_path)

    def dataset_overview(self):
        """Display basic information and the first few rows of the dataset."""
        print(self.data.info())
        print(self.data.head())

    def visualize_distributions(self):
        """Visualize the distribution of each feature in the dataset."""
        for col in self.data.columns:
            plt.figure(figsize=(8, 4))
            sns.countplot(x=self.data[col])
            plt.title(f"Distribution of {col}")
            plt.xticks(rotation=45)
            plt.show()

    def plot_correlation_heatmap(self):
        """Plot a heatmap to show correlations between features."""
        encoded_data = self.data.apply(lambda x: pd.factorize(x)[0])
        correlation_matrix = encoded_data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap for Categorical Data")
        plt.show()

    def visualize_maintenance_distribution(self):
        """Visualize the distribution of the maintenance cost."""
        plt.figure(figsize=(8, 4))
        sns.countplot(x=self.data["maint"], palette="Set2")
        plt.title("Distribution of Maintenance Cost")
        plt.show()

    def visualize_feature_relationships(self):
        """Show count plots to explore relationships between features and maintenance cost."""
        features = ["buying", "doors", "persons", "lug_boot", "safety"]
        for feature in features:
            plt.figure(figsize=(8, 4))
            sns.countplot(x=self.data[feature], hue=self.data["maint"], palette="Set3")
            plt.title(f"Relationship between {feature} and Maintenance Cost")
            plt.show()

    def plot_pairwise_relationships(self):
        """Plot pairwise relationships using pairplot."""
        encoded_data = self.data.apply(LabelEncoder().fit_transform)
        sns.pairplot(encoded_data, hue="maint", palette="Set2")
        plt.show()

    def plot_feature_importances(self):
        """Display feature importances from a RandomForestClassifier."""
        X = self.data.apply(LabelEncoder().fit_transform).drop("maint", axis=1)
        y = self.data.apply(LabelEncoder().fit_transform)["maint"]
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X, y)
        importances = rf.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]
        plt.title("Feature Importance")
        plt.bar(range(X.shape[1]), importances[sorted_indices], align="center")
        plt.xticks(range(X.shape[1]), X.columns[sorted_indices], rotation=90)
        plt.show()

    def plot_boxplots_against_maint(self):
        """Create box plots for each feature against encoded maintenance cost."""
        self.data["maint_encoded"] = LabelEncoder().fit_transform(self.data["maint"])
        for feature in ["buying", "doors", "persons", "lug_boot", "safety"]:
            plt.figure(figsize=(8, 4))
            sns.boxplot(x=self.data[feature], y=self.data["maint_encoded"])
            plt.title(f"{feature} vs Maintenance Cost")
            plt.show()

    def plot_facetgrid(self):
        """Generate a FacetGrid to explore relationships between multiple features."""
        g = sns.FacetGrid(
            self.data,
            col="doors",
            row="safety",
            hue="maint",
            palette="Set2",
            margin_titles=True,
        )
        g.map(sns.scatterplot, "buying", "persons")
        g.add_legend()
        plt.show()

    def perform_pca_and_plot(self):
        """Apply PCA to reduce dimensionality and plot the results."""
        encoded_data = self.data.apply(LabelEncoder().fit_transform)
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(encoded_data)
        plt.scatter(
            reduced_data[:, 0],
            reduced_data[:, 1],
            c=encoded_data["maint"],
            cmap="viridis",
        )
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.colorbar(label="Maintenance Cost")
        plt.show()

    def plot_stacked_bar_charts(self):
        """Plot stacked bar charts to show maintenance cost distribution across features."""
        for feature in ["buying", "doors", "persons", "lug_boot"]:
            pd.crosstab(self.data[feature], self.data["maint"]).plot(
                kind="bar", stacked=True
            )
            plt.title(f"Stacked Bar Chart: {feature} vs Maintenance Cost")
            plt.show()
