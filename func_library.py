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
          "comp_total_metrics": {
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
    - comp_total_metrics: Dictionary of expected descriptive statistics for the 'CompTotal' column.
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
        assert os.path.exists(test_metrics_filepath), f"File {test_metrics_filepath} does not exist" # asserts from __init__ should be moved to perform_tests to prevent confusion about exception origins. Oh well.

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

    
class PPPDataTester:
    """
    Class to test the OECD Purchasing Power Parity (PPP) data. This class is designed to verify the integrity and accuracy of PPP data against expected metrics provided in a JSON file.
    
    The class is initialized with the path to the PPP data file and the path to a JSON file containing the test metrics. It performs several checks, including dimensions of the data, count of missing values, and sums of values for specified years.
    
    Initialization Parameters:
    - data_filepath: Path to the CSV file containing PPP data.
    - test_metrics_filepath: Path to the JSON file containing the test metrics.
    
    JSON File Schema:
    {
      "PPP": {
        "row_count": <int>,
        "column_count": <int>,
        "missing_vals_count": <int>,
        "year_sums": {
          "2019": <float>,
          "2020": <float>,
          "2021": <float>,
          "2022": <float>,
          "2023": <float>
        }
      }
    }
    
    The JSON file must contain:
    - row_count: Expected total number of rows in the PPP data.
    - column_count: Expected total number of columns in the PPP data.
    - missing_vals_count: Expected total count of missing (NaN) values across all columns.
    - year_sums: A dictionary where each key is a year of interest and its corresponding value is the expected sum of PPP values for that year.
    
    The class methods include checks for the data's dimensions, the total count of missing values, and the sums of PPP values for specified years against the expectations defined in the JSON file.
    """

    def __init__(self, data_filepath: str = r"data/ppp.csv", test_metrics_filepath: str = r"data/data_test_metrics.json"):
        """
        The class is initialized with the path to the PPP data file and the path to a JSON file containing the test metrics.
        """
        self.years_of_interest = ["2019", "2020", "2021", "2022"]
        self.test_metrics_filepath = test_metrics_filepath
        
        # test existence of metrics file with pytest
        assert os.path.exists(test_metrics_filepath), f"File {test_metrics_filepath} does not exist"

        # test existence of file with pytest
        assert os.path.exists(data_filepath), f"File {data_filepath} does not exist"
        self.df = pd.read_csv(data_filepath, header=2)

        #read json file into dictionary
        metrics_dict = {}
        with open(test_metrics_filepath, "r") as file:
            metrics_contents = json.load(file)
            metrics_dict = metrics_contents["PPP"]

        self.expected_row_count = metrics_dict["row_count"]
        self.expected_column_count = metrics_dict["column_count"]
        self.expected_missing_vals_count = metrics_dict["missing_vals_count"]
        self.expected_year_sums = metrics_dict["year_sums"]

    def _test_dims(self):
        rows, cols = self.df.shape
        assert rows == self.expected_row_count, f"Row count does not match. Expected {self.expected_row_count}, got {rows}"
        assert cols == self.expected_column_count, f"Column count does not match. Expected {self.expected_column_count}, got {cols}"
    
    def _test_nan_count(self):
        nan_count = self.df.isna().sum().sum()
        assert nan_count == self.expected_missing_vals_count, f"Missing values count does not match. Expected {self.expected_missing_vals_count}, got {nan_count}"
    
    def _test_year_sums(self):
        for year, expected_sum in self.expected_year_sums.items():
            actual_sum = self.df[year].sum()
            assert actual_sum == expected_sum, f"Yearly sum does not match. Expected {expected_sum}, got {actual_sum}"
    
    def perform_tests(self):
        self._test_dims()
        self._test_nan_count()
        self._test_year_sums()
        print(f"All tests passed for PPP data.")
    
    @staticmethod
    def update_ppp_metrics_json(ppp_data_location, data_metrics_json_path):
        """
        Update the JSON file with test metrics for the OECD PPP data.
        """

        try:
            if os.path.exists(data_metrics_json_path) and os.path.getsize(data_metrics_json_path) > 0:
                with open(data_metrics_json_path, 'r') as json_file:
                    metrics = json.load(json_file)
            else:
                raise FileNotFoundError  # Trigger the except block to initialize metrics
        except (json.JSONDecodeError, FileNotFoundError):
            metrics = {"PPP": {}}
            print("Existing JSON file not found or malformed; initializing new structure.")
        
        # Assuming the PPP data file structure is consistent across years
        ppp_filename = ppp_data_location
        if not os.path.exists(ppp_filename):
            raise FileNotFoundError(f"File {ppp_filename} does not exist")

        df = pd.read_csv(ppp_filename, header=2)
        
        row_count = len(df)
        column_count = len(df.columns)
        missing_vals_count = int(df.isna().sum().sum())

        # Calculate sums for years of interest
        year_sums = {}
        for year in ["2019", "2020", "2021", "2022"]:
            if year in df.columns:
                year_sums[year] = float(df[year].sum())
            else:
                year_sums[year] = None  # or 0, depending on how you want to handle missing years
        
        metrics["PPP"] = {
            "row_count": row_count,
            "column_count": column_count,
            "missing_vals_count": missing_vals_count,
            "year_sums": year_sums
        }

        try:
            json.dumps(metrics)  # Quick validation before writing
        except TypeError as e:
            print(f"Error serializing the metrics: {e}")
            return

        with open(data_metrics_json_path, 'w') as json_file:
            json.dump(metrics, json_file, indent=4)
        
        print(f"Updated metrics JSON file has been saved to {data_metrics_json_path}.")


class AIJobsDataTester:
    """
    Class to test the AI Jobs data. This class is designed to verify the integrity and accuracy of AI Jobs data against expected metrics provided in a JSON file.
    
    The class is initialized with the path to the AI Jobs data file and the path to a JSON file containing the test metrics. It performs several checks, including dimensions of the data, count of missing values, and statistical analysis of the salary data.
    
    Initialization Parameters:
    - data_filepath: Path to the CSV file containing AI Jobs data.
    - test_metrics_filepath: Path to the JSON file containing the test metrics.
    
    JSON File Schema:
    {
      "AIJobs": {
        "row_count": <int>,
        "column_count": <int>,
        "missing_vals_count": <int>,
        "salary_stats": {
          "mean": <float>,
          "std": <float>,
          "min": <float>,
          "25%": <float>,
          "50%": <float>,
          "75%": <float>,
          "max": <float>
        }
      }
    }
    
    The JSON file must contain:
    - row_count: Expected total number of rows in the AI Jobs data.
    - column_count: Expected total number of columns in the AI Jobs data.
    - missing_vals_count: Expected total count of missing (NaN) values across all columns.
    - salary_stats: A dictionary of expected descriptive statistics for the 'salary' column, including mean, standard deviation, minimum, quartiles, and maximum values.
    
    The class methods include checks for the data's dimensions, the total count of missing values, and statistical analysis of the salary column against the expectations defined in the JSON file.
    """

    def __init__(self, data_filepath: str = r"data/ai-jobs_salaries.csv", test_metrics_filepath: str = r"data/data_test_metrics.json"):
        """
        The class is initialized with the path to the AI Jobs data file and the path to a JSON file containing the test metrics.
        """
        self.test_metrics_filepath = test_metrics_filepath
        
        # test existence of metrics file with pytest
        assert os.path.exists(test_metrics_filepath), f"File {test_metrics_filepath} does not exist"

        # test existence of file with pytest
        assert os.path.exists(data_filepath), f"File {data_filepath} does not exist"
        self.df = pd.read_csv(data_filepath)

        #read json file into dictionary
        metrics_dict = {}
        with open(test_metrics_filepath, "r") as file:
            metrics_contents = json.load(file)
            metrics_dict = metrics_contents["AIJobs"]

        self.expected_row_count = metrics_dict["row_count"]
        self.expected_column_count = metrics_dict["column_count"]
        self.expected_missing_vals_count = metrics_dict["missing_vals_count"]
        self.expected_salary_stats = metrics_dict["salary_stats"]

    def _test_dims(self):
        rows, cols = self.df.shape
        assert rows == self.expected_row_count, f"Row count does not match. Expected {self.expected_row_count}, got {rows}"
        assert cols == self.expected_column_count, f"Column count does not match. Expected {self.expected_column_count}, got {cols}"

    def _test_nan_count(self):
        nan_count = self.df.isna().sum().sum()
        assert nan_count == self.expected_missing_vals_count, f"Missing values count does not match. Expected {self.expected_missing_vals_count}, got {nan_count}"
    
    def _test_salary_stats(self):
        salary_stats = self.df["salary"].describe()

        stats_to_check = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
        for stat in stats_to_check:
            assert salary_stats[stat] == self.expected_salary_stats[stat], f"Salary {stat} does not match. Expected {self.expected_salary_stats[stat]}, got {salary_stats[stat]}"

    def perform_tests(self):
        self._test_dims()
        self._test_nan_count()
        self._test_salary_stats()
        print(f"All tests passed for AI Jobs data.")

    @staticmethod
    def update_ai_jobs_metrics_json(ai_jobs_data_location, data_metrics_json_path):
        """
        Static method to update the JSON file with test metrics for the AI Jobs data. The method reads the AI Jobs data file, computes necessary metrics, and updates or creates a JSON file with these metrics under an "AIJobs" key.
        
        Parameters:
        - ai_jobs_data_location: The file path to the AI Jobs data file.
        - data_metrics_json_path: The file path to the JSON file where the metrics will be stored.
        """

        try:
            if os.path.exists(data_metrics_json_path) and os.path.getsize(data_metrics_json_path) > 0:
                with open(data_metrics_json_path, 'r') as json_file:
                    metrics = json.load(json_file)
            else:
                raise FileNotFoundError  # Trigger the except block to initialize metrics
        except (json.JSONDecodeError, FileNotFoundError):
            metrics = {"AIJobs": {}}
            print("Existing JSON file not found or malformed; initializing new structure.")
        
        # Assuming the AI Jobs data file structure is consistent
        ai_jobs_filename = ai_jobs_data_location
        if not os.path.exists(ai_jobs_filename):
            raise FileNotFoundError(f"File {ai_jobs_filename} does not exist")

        df = pd.read_csv(ai_jobs_filename)
        
        row_count, column_count = df.shape
        missing_vals_count = int(df.isna().sum().sum())
        salary_stats = df["salary"].describe().to_dict()
        # Convert NumPy types to Python native types for JSON serialization
        salary_stats = {key: float(value) for key, value in salary_stats.items()}
        
        metrics["AIJobs"] = {
            "row_count": row_count,
            "column_count": column_count,
            "missing_vals_count": missing_vals_count,
            "salary_stats": salary_stats
        }

        try:
            json.dumps(metrics)  # Quick validation before writing
        except TypeError as e:
            print(f"Error serializing the metrics: {e}")
            return

        with open(data_metrics_json_path, 'w') as json_file:
            json.dump(metrics, json_file, indent=4)
        
        print(f"Updated metrics JSON file has been saved to {data_metrics_json_path}.")