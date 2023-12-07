class CarDataset:
    def __init__(self, file_path):
        """
        Initialize the CarDataset class with the file path of the dataset.
        """
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """
        Load the dataset from the CSV file.
        """
        try:
            self.data = pd.read_csv(self.file_path)
            print("Dataset loaded successfully.")
        except Exception as e:
            print(f"Error loading dataset: {e}")

    def display_head(self, n=5):
        """
        Display the first n rows of the dataset.
        """
        if self.data is not None:
            return self.data.head(n)
        else:
            print("Dataset not loaded. Please load the dataset first.")

    def dataset_info(self):
        """
        Print information about the dataset including column names, data types, and missing values.
        """
        if self.data is not None:
            print(self.data.info())
        else:
            print("Dataset not loaded. Please load the dataset first.")

    def summary_statistics(self):
        """
        Display summary statistics of the dataset.
        """
        if self.data is not None:
            return self.data.describe(include="all")
        else:
            print("Dataset not loaded. Please load the dataset first.")
