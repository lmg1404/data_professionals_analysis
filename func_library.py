import json
import os
import pytest
import pandas as pd



def get_SO_dataframes(data_location: str) -> dict:
    """
    Returns a dictionary of dataframes for each year of Stack Overflow survey data
    """
    SO_dataframes = {}
    for year in range(2020, 2023):
        year = str(year)
        df = pd.read_csv(f"{data_location}/survey_results_public_{year}.csv")
        SO_dataframes[year] = {"df": df}
    return SO_dataframes


class StackOverflowDataTester:
    """
    Class to test the Stack Overflow survey data. Requires a corresponding JSON file with test metrics for the given survey year.
    The class is initialized with the survey year and the location of the data. It compares the metrics in the JSON file with the actual data to determine the accuracy of the data.
    
    JSON File Schema:
    {
      "StackOverflow": {
        "<survey_year>": {
          "row_count": <int>,
          "column_count": <int>,
          "column_names": [<list_of_string>],
          "comp_total_total": {
            "mean": <float>,
            "std": <float>,
            "min": <float>,
            "25%": <float>,
            "50%": <float>,
            "75%": <float>,
            "max": <float>
          },
          "missing_vals_count": <int>
        },
        ... (additional years as necessary)
      }
    }

    Each <survey_year> key contains metrics specific to that year's survey data, including:
    - row_count: Expected total number of rows.
    - column_count: Expected total number of columns.
    - column_names: List of expected column names.
    - comp_total_total: Statistical summary for the 'CompTotal' column, including mean, std, min, quartiles, and max.
    - unique_country_count: Expected number of unique countries.
    - missing_vals_count: Expected total count of missing (NaN) values across all columns.
    """

    def __init__(self, survey_year: str, data_location: str = r"data/stack_overflow", test_metrics_filepath: str = r"data/data_test_metrics.json"):
        """
        The class is initialized with the survey year and the location of the data
        """
        self.csv_filename_prefix = "survey_results_public_"
        self.survey_year = survey_year
        self.data_location = data_location
        
        # test existence of metrics file with pytest
        assert os.path.exists(test_metrics_filepath), f"File {test_metrics_filepath} does not exist"

        # test existence of file with pytest
        filepath = f"{data_location}/{self.csv_filename_prefix}{survey_year}.csv" # eg. "../stack_overflow_data/survey_results_public_2020.csv"
        assert os.path.exists(filepath), f"File {filepath} does not exist"
        self.df = pd.read_csv(filepath)
        
        #read json file into dictionary
        metrics_dict = {}
        with open(test_metrics_filepath, "r") as file:
            metrics_contents = json.load(file)
            metrics_dict = metrics_contents["StackOverflow"][survey_year]

        self.row_count = metrics_dict["row_count"]
        self.column_count = metrics_dict["column_count"]
        self.expected_column_names = metrics_dict["column_names"]
        self.comp_total_metrics = metrics_dict["comp_total_metrics"]
        self.expected_missing_vals_count = metrics_dict["missing_vals_count"]

    def _test_dims(self):
        rows, cols = self.df.shape
        assert rows == self.row_count, f"Row count does not match. Expected {self.row_count}, got {rows}"
        assert cols == self.column_count, f"Column count does not match. Expected {self.column_count}, got {cols}"

    def _test_column_names(self):
        actual_col_names = set([str(c)for c in self.df.columns.to_list()])
        expected_col_names = set(self.expected_column_names)
        
        missing_cols = expected_col_names - actual_col_names
        unexpected_cols = actual_col_names - expected_col_names
        
        assert not missing_cols, f"Missing expected columns: {missing_cols}"
        assert not unexpected_cols, f"Found unexpected columns: {unexpected_cols}"

    def _test_nan_count(self):
        nan_count = self.df.isna().sum().sum()
        assert nan_count == self.expected_missing_vals_count, f"Missing values count does not match. Expected {self.expected_missing_vals_count}, got {nan_count}"

    def _test_comp_total_stats(self):
        comp_total_stats = self.df["CompTotal"].describe()

        stats_to_check = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
        for stat in stats_to_check:
            assert comp_total_stats[stat] == self.comp_total_metrics[stat], f"CompTotal {stat} does not match. Expected {self.comp_total_metrics[stat]}, got {comp_total_stats[stat]}"

    def perform_tests(self):
        self._test_dims()
        self._test_column_names()
        self._test_nan_count()
        self._test_comp_total_stats()
        print(f"All tests passed for {self.survey_year} survey data.")

    @staticmethod
    def update_test_metrics_json(stack_overflow_data_location, data_metrics_json_path):
        """
        Update the JSON file with test metrics for the Stack Overflow survey data.
        """

        try:
            if os.path.exists(data_metrics_json_path) and os.path.getsize(data_metrics_json_path) > 0:
                with open(data_metrics_json_path, 'r') as json_file:
                    metrics = json.load(json_file)
            else:
                raise FileNotFoundError  # Trigger the except block to initialize metrics
        except (json.JSONDecodeError, FileNotFoundError):
            metrics = {"StackOverflow": {}}
            print("Existing JSON file not found or malformed; initializing new structure.")
        
        for year in range(2019, 2024):
            csv_filename = f"{stack_overflow_data_location}/survey_results_public_{year}.csv"
            if not os.path.exists(csv_filename):
                raise FileNotFoundError(f"File {csv_filename} does not exist")

            df = pd.read_csv(csv_filename)
            
            # Ensure CompTotal exists in the DataFrame
            if 'CompTotal' in df.columns:
                comp_total_stats = df["CompTotal"].describe().to_dict()
                # Convert NumPy types to Python native types for JSON serialization
                comp_total_stats = {key: float(value) for key, value in comp_total_stats.items()}
            else:
                comp_total_stats = {}
                
            row_count = len(df)
            column_count = len(df.columns)
            column_names = [str(c) for c in list(df.columns)]
            missing_vals_count = int(df.isna().sum().sum())
            
            metrics["StackOverflow"][str(year)] = {
                "row_count": row_count,
                "column_count": column_count,
                "column_names": column_names,
                "comp_total_metrics": comp_total_stats,
                "missing_vals_count": missing_vals_count
            }

        try:
            json.dumps(metrics)  # Quick validation before writing
        except TypeError as e:
            print(f"Error serializing the metrics: {e}")
            return

        with open(data_metrics_json_path, 'w') as json_file:
            json.dump(metrics, json_file, indent=4)
        
        print(f"Updated metrics JSON file has been saved to {data_metrics_json_path}.")

    


