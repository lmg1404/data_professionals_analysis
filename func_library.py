import json
import os
import pytest
import pycountry
import numpy as np
import pandas as pd
from rapidfuzz import process
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer

YEARS_OF_INTEREST = ["2019", "2020", "2021", "2022", "2023"]

def fuzzy_country_match(name, countries_list, threshold_distance=80):
    """
    Uses fuzzy matching to find the closest country name to the given name
    Parameters:
    - name (str): The name to match
    - countries_list (list): A list of country names to match against
    - threshold_distance (int): The minimum similarity score required to consider a match
    Returns:
    - str: The closest matching country name, or None if no match is found
    """
    closest_match = process.extractOne(name, countries_list)
    # if the closest match is less than 80% similar, return None
    if closest_match and closest_match[1] < threshold_distance:
        return None
    
    return closest_match[0] if closest_match else None

def get_country_code_from_name(country_name, countries):
    """
    Returns the ISO 3166-1 alpha-2 code for the given country name
    """
    try:
        name_adjustment_map = {
            "Turkey":"Türkiye"
        }
        country_name = name_adjustment_map[country_name] if country_name in name_adjustment_map.keys() else country_name
        return pycountry.countries.lookup(country_name).alpha_2
    except LookupError:
        # Try fuzzy matching
        closest_name = fuzzy_country_match(country_name, countries)
        if closest_name:
            return pycountry.countries.lookup(closest_name).alpha_2
        else:
            return None

def get_raw_SO_dataframes(data_location: str) -> dict:
    """
    Returns a dictionary of raw dataframes for each year of Stack Overflow survey data
    """
    SO_dataframes = {}
    for year in range(2020, 2023):
        year = str(year)
        df = pd.read_csv(f"{data_location}/survey_results_public_{year}.csv")
        SO_dataframes[year] = {"df": df}
    return SO_dataframes

def adjust_usd_to_2023_usd(old_usd: float, year: str) -> float:
    """
    Adjusts an amount of USD from a given year to its equivalent value in
    December 2023 USD using the Consumer Price Index (CPI) inflation factor
    for the given year. The CPI factors were retrieved from the US Bureau of
    Labor Statistics website on February 1, 2024:
    https://www.bls.gov/data/inflation_calculator.htm

    Parameters:
    - old_usd (float): The amount of USD to adjust.
    - year (str): The year from which the amount is to be adjusted. Must be
      a string representing a year for which the CPI inflation factor is known.

    Returns:
    - float: The adjusted amount in December 2023 USD.
    """
    cpi_inflation_factors = {
        "2017": 1.24,
        "2018": 1.22,
        "2019": 1.19,
        "2020": 1.18,
        "2021": 1.10,
        "2022": 1.03,
        "2023": 1.00
    }

    return old_usd * cpi_inflation_factors[year]


def read_ppp(csv_filepath="data/imf-dm-export-20240204.csv"):
    """
    creates a dataframe of the ppp factors
    :param csv_filepath: the file path to the csv file
    """
    imf_ppp_df = pd.read_csv(csv_filepath)
    imf_ppp_df = imf_ppp_df.drop(imf_ppp_df.index[0])
    
    # remove year columns that are not in years_of_interest]
    cols_to_drop = [col for col in imf_ppp_df.columns if (col not in YEARS_OF_INTEREST) and (col.startswith("1") or col.startswith("2"))]
    imf_ppp_df = imf_ppp_df.drop(columns=cols_to_drop)
    
    # create country column with ISO 3166-1 alpha-3 country codes
    imf_ppp_df = imf_ppp_df.rename(columns={imf_ppp_df.columns[0]: 'country_full_name'})
    country_list = [country.name for country in pycountry.countries]
    get_country_code = lambda c: get_country_code_from_name(country_name=c, countries=country_list)
    imf_ppp_df['country'] = imf_ppp_df['country_full_name'].apply(get_country_code)
    return imf_ppp_df


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
        self.years_of_interest = YEARS_OF_INTEREST
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
        for year in YEARS_OF_INTEREST:
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


class StackOverflowData:
    """
    This class is used to hold the Stack Overflow data and perform various operations on it.
    """

    @staticmethod
    def generate_aggregate_df(only_data_science_devs=True) -> (pd.DataFrame, list, list):
        """
        Reads CSVs and gets the numbe of data professionals. Any empty values are dropped from job title and 
        salary so we will always have data. Other columns may have nans.
        Data Manipulation:
        - dropping nans from salary and devtype combined
        - Changing the salary column to ConvertedCompYearly so we can merge all data frames comes time
        - Lowering column names since there was some weird camel case going on
        - Converting specific columns that mean the same thing per year into a singular name
        - Fill in nans for language/skill specific values with "nan"
        - this is so we can one hot later on for a more concise analysis, more later on
        - Binarize the different skills per year, see create_onehot_skills
        - Next we abbreviate education levels so that we can also one hot them, see above
        - Change education to keep into one column binarizing doesn't make any sense
        - Changing org size into something much more manageable, mainly the I don't know field
        - We merge them into one, the same groupby operations can still be done as if seperate
        - Encode devtype to binarize as well since it's very difficult to parse through ; every single time
            - we can do some clever work arounds
        - Lastly we return the skills columns really quick to save headache later on


        Input: on
        Outputs: tuple(pd.DataFrame, list[str], list[str])
        """


        frames = {}
        stack_o_files = os.listdir("data/stack_overflow/")
        for file in stack_o_files:
            year = file[-8:-4]
            df = pd.read_csv(f"data/stack_overflow/{file}", encoding='ISO-8859-1')

            # standardize compensation columns
            if 'ConvertedComp' in df.columns:
                df = df.rename(columns={'ConvertedComp': 'ConvertedCompYearly'})

            # standardize some columns
            # using camel case resulted in errors with webframe where sometimes F was capitalized
            standard = ["language", "database", "platform", "webframe", "misctech"]
            df.columns = df.columns.str.lower()
            for stan in standard:
                if f"{stan}workedwith" in df.columns:
                    df = df.rename(columns={f'{stan}workedwith': f'{stan}haveworkedwith', f'{stan}desirenextyear':f'{stan}wanttoworkwith'})
                df[f"{stan}haveworkedwith"] = df[f"{stan}haveworkedwith"].fillna(value="Empty")
                df[f"{stan}wanttoworkwith"] = df[f"{stan}wanttoworkwith"].fillna(value="Empty")

            # standardize some country names, now they should match with Kaggle dataset
            df['original_country'] = df['country'].copy()
            country_list = [country.name for country in pycountry.countries]
            get_country_code = lambda c: get_country_code_from_name(country_name=c, countries=country_list)
            df["country"] = df["country"].apply(get_country_code)

            # we have some numbers so we can't just do entire df
            df[['edlevel', 'orgsize']] = df[['edlevel', 'orgsize']].fillna(value="nan")
            df['orgsize'] = df['orgsize'].replace({'I donâ\x80\x99t know': 'IDK'})
            
            df = df.dropna(subset=["devtype", "convertedcompyearly"])
            if only_data_science_devs:
                df = df[df["devtype"].str.contains("data", case=False)]
            df["count"] = [1] * len(df) # this is for our groupby so that we can say count > cull when we sum or count
            df["year"] = [year] * len(df)
            frames[f"df_data_{year}"] = df

        # oops forgot indentation
        StackOverflowData.abbr_education(frames)
        StackOverflowData.bin_ages(frames)
        StackOverflowData.create_onehot_skills(frames)
        similar = StackOverflowData.find_similar_col(frames)
        
        # finally going to standardize to merge devtypes
        for key, frame in frames.items():
            frames[key] = frame[similar]
        df = pd.concat([frame for key, frame in frames.items()], axis=0)
        df, employment = StackOverflowData.encode_devtype(df)
        skills = [col for col in df.columns if any(substr in col for substr in ['lg', 'db', 'pf', 'wf', 'mt'])]
        
        return df, skills, employment

    @staticmethod
    def create_onehot_skills(frames: dict) -> None:
        """
        Given a dictionary of pandas dataframes we want to one hot the skills in particular.
        We want to take the skills in the different columns and one hot them such we can sum them for groupby operations.
        We get a dictionary of pandas DataFrames and perform an inplace operation such that we don't have to create new memory.
        Return a dictionary of a list of strings for a couple reasons:
            - there's no way we will remember all of these so automation by putting these into a list seemed like the best idea
            - the keys will match those in the input in case we want to do something with these later per year
            - hashing onto a dictionary should allow for ease of access since no 2 years will have the same EXACT one hot columns, hence the list
        The above is deprecated, after merging with similar columns these will all be useless to us

        We also drop the _Empty for EVERYTHING since that information is useless to us
        
        Input: frames dict{str: pd.DataFrames}
        Ouput: None

        https://stackoverflow.com/questions/45312377/how-to-one-hot-encode-from-a-pandas-column-containing-a-list

        Rough example flow of function for one sample:
        C; C++; Perl -> [C, C++, Perl] -> [1, 1, 1, 0]
        Python       -> [Python]       -> [0, 0, 0, 1]
        """
        # some constants
        standard = [("language", "lg"), ("database", "db"), ("platform", "pf"), ("webframe", "wf"), ("misctech", "mt")]
        status = [("wanttoworkwith", "www"), ("haveworkedwith", "hww")]
        
        for key, frame in frames.items():
            new_cols = []
            for stan, abv in standard:
                for stat, abr in status:
                    coi = stan + stat # coi = column of interest
                    abbr = abv + abr + "_"
                    mlb = MultiLabelBinarizer(sparse_output=True) # saves ram
                    frame[coi] = frame[coi].str.split(";")
                    transformed = mlb.fit_transform(frame.pop(coi))
                    new_cois = [abbr + name for name in mlb.classes_]
                    frame = frame.join(
                                pd.DataFrame.sparse.from_spmatrix(
                                    transformed,
                                    index=frame.index,
                                    columns=new_cois
                                )
                            )
                    new_cois.remove(abbr + "Empty")
                    new_cols += new_cois
                    frame = frame.drop(abbr + "Empty", axis=1)
            # this needs to be here, if not throse Sparse type errors
            # # Sparse types don't allow normal groupby operations (ie reshape) so we need to turn them into ints
            # # int8 don't take up a ton and it's just 0's and 1's
            # # for all intents and purposes these are sparse matrices, we just want to avoid the object
            frame[new_cols] = frame[new_cols].fillna(0)
            frame[new_cols] = frame[new_cols].astype('int8')
            frames[key] = frame

    @staticmethod
    def bin_ages(frames: dict) -> None:
        """
        Map ages to bins and then to strings for uniformity across years and to make it easier for one hot encoding
        """
        bins = [0, 18, 24, 34, 44, 54, 64, 100]
        labels = ['Under 18 years old', '18-24 years old', '25-34 years old', '35-44 years old', '45-54 years old', '55-64 years old', '65 years or older']
        for year, frame in frames.items():    
            if frame["age"].dtypes == float:
                frame["age"] = pd.cut(frame["age"], bins=bins, labels=labels)
            frame["age"] = frame["age"].astype('str')
            
        frames[year] = frame

    @staticmethod
    def abbr_education(frames: dict) -> None:
        """
        Similar in spirit to the other one hots, but this is in place
        Automatically abbreviates education levels across all frames
        Had to hard code the list again, not a big deal only 8 items
        
        Input: frames dict{str: pd.DataFrames}
        Ouput: None
        """
        # more hardcoded stuff that are needed
        abbreviations = ["Associate's", "Bachelor's", "Master's", "Elementary", "Professional", "Secondary", "Some College", "Else"]
        
        for key, frame in frames.items():
            # easier to replace this, makes it much easier to work with
            frame['edlevel'] = frame['edlevel'].replace({'I never completed any formal education': 'Something else'})

            # need the sorted since they have the same rough scheme
            levels = list(frame['edlevel'].unique())
            levels.sort()
            o = 0 # offset

            # dictionary to feed into repalce function
            replace_dict = {}
            for i in range(len(levels)):
                col = levels[i]
                if col == 'nan':
                    break
                abbr = abbreviations[i-o]
                if 'doctoral' in col:
                    replace_dict[col] = "Doctoral"
                    o += 1
                    continue
                replace_dict[col] = abbr
                    
            frame['edlevel'] = frame['edlevel'].replace(replace_dict)
            frames[key] = frame

    @staticmethod
    def find_similar_col(frames) -> list:
        """
        Returns the set of columns that the all share, ideally we maximize the ratio of this to merge.
        """
        union = []
        for key, frame in frames.items():
            union.append(set(frame.columns))
            
        standard = union[0]
        for cols in union[1:]:
            standard = standard.intersection(cols)
        return list(standard)

    @staticmethod
    def encode_devtype(df: pd.DataFrame) -> (pd.DataFrame, list):
        """
        Standardizing DevType so that we can merge on ai-net data
        """
        def map_job(category_list) -> list:
            devtype = set()
            for category in category_list:
                if (clean := "data scientist") in category.lower():
                    devtype.add(clean)
                elif "math" in category.lower() or "stat" in category.lower():
                    devtype.add("mathematician_statistician")
                elif (clean := "analyst") in category.lower():
                    devtype.add(clean)
                elif (clean := "manage") in category.lower():
                    devtype.add(clean + "ment")
                elif (clean := "scientist") in category.lower():
                    devtype.add(clean + "_other")
                elif (clean := "engineer") in category.lower():
                    devtype.add(clean + "_other")
                elif (clean := "developer") in category.lower():
                    devtype.add(clean)
                else:
                    devtype.add("systems_architect")
            return list(devtype)
            
        coi = "devtype"
        mlb = MultiLabelBinarizer(sparse_output=True) # saves ram
        df[coi] = df[coi].str.split(";")
        df[coi] = df[coi].apply(map_job)
        transformed = mlb.fit_transform(df.pop(coi))
        new_cols = mlb.classes_
        df = df.join(
                    pd.DataFrame.sparse.from_spmatrix(
                        transformed,
                        index=df.index,
                        columns=mlb.classes_
                    )
                )
        # see above binarizer
        df[new_cols] = df[new_cols].fillna(0)
        df[new_cols] = df[new_cols].astype('int8')
        return df, new_cols
    


def adjust_usd_to_2023_usd(old_usd: float, year: str) -> float:
    """
    Adjusts an amount of USD from a given year to its equivalent value in
    December 2023 USD using the Consumer Price Index (CPI) inflation factor
    for the given year. The CPI factors were retrieved from the US Bureau of
    Labor Statistics website on February 1, 2024:
    https://www.bls.gov/data/inflation_calculator.htm

    Parameters:
    - old_usd (float): The amount of USD to adjust.
    - year (str): The year from which the amount is to be adjusted. Must be
      a string representing a year for which the CPI inflation factor is known.

    Returns:
    - float: The adjusted amount in December 2023 USD.
    """
    cpi_inflation_factors = {
        "2017": 1.24,
        "2018": 1.22,
        "2019": 1.19,
        "2020": 1.18,
        "2021": 1.10,
        "2022": 1.03,
        "2023": 1.00
    }

    return old_usd * cpi_inflation_factors[year]


class AISalariesData:
    """
    Class to hold the AI Salaries data and perform various operations on it.
    """
    
    @staticmethod
    def generate_df(csv_filepath: str = r"data/ai-jobs_salaries.csv") -> pd.DataFrame:
        """
        Reads the salaries from ai-net and returns them into a dataframe
        Data Manipulation:
        - Change 2 letter country names into 3 letter names for uniformity
        - Map above function in job_title to simpler names
        - Only taking 2020 - 2023, we have no data on 2024
        
        Input: None
        Output: pd.DataFrame
        """
        salaries = pd.read_csv(csv_filepath)
        country_abbreviations = {country.alpha_2: country.alpha_3 for country in pycountry.countries}
        mapping = AISalariesData.process_job_titles(salaries)
        
        salaries[["employee_residence", "company_location"]] = salaries[["employee_residence", "company_location"]].replace(country_abbreviations)
        salaries["job_title"] = salaries["job_title"].replace(mapping)
        salaries = salaries[salaries["work_year"] < 2024]
        
        return salaries
    
    @staticmethod
    def process_job_titles(salaries_df: pd.DataFrame) -> dict:
        """
        Helper function that is just a for loop that goes through unique job titles and assigns a basic name

        Input: pd.DataFrame
        Output: dict{str: str}
        """
        mapping = {}
        for job in list(salaries_df["job_title"].unique()):
            if (short := "Analyst") in job:
                mapping[job] = short.lower()
        
            elif (short := "Engineer") in job:
                mapping[job] = short.lower() + "_other"
                
            elif (short := "Data Scientist") in job or "Data Science" in job:
                    mapping[job] = '_'.join(short.lower().split(" "))
                
            elif "Architect" in job:
                mapping[job] = "systems_architect"
        
            elif "Manager" in job:
                mapping[job] = "management"
        
            elif (short := "Developer") in job:
                mapping[job] = short.lower()
                
            elif "math" in job.lower() or "stat" in job.lower():
                mapping[job] = "mathematician_statistician"
                
            else:
                mapping[job] = "scientist_other"
        return mapping
    

if __name__ == "__main__":
    stack_overflow, skills_list, employments = StackOverflowData.generate_aggregate_df(only_data_science_devs=True)
    stack_overflow.shape
    ppp_df = read_ppp()
    print("completed")