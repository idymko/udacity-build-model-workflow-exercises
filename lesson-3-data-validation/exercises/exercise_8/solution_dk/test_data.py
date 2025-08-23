import pytest
import wandb
import pandas as pd
import scipy.stats

# This is global so all tests are collected under the same
# run
run = wandb.init(project="exercise_8", job_type="data_tests")


@pytest.fixture(scope="session")
# "session": The fixture is created once per test session (useful for expensive setup).
# "function" (default): The fixture is created and destroyed for each test function.
def data():

    local_path = run.use_artifact("exercise_6/data_train.csv:latest").file()
    sample1 = pd.read_csv(local_path)

    local_path = run.use_artifact("exercise_6/data_test.csv:latest").file()
    sample2 = pd.read_csv(local_path)
    
    # sample1.to_csv('sample1.csv')
    # sample2.to_csv('sample2.csv')

    return sample1, sample2

def test_nan(data):
    
    sample1, sample2 = data

    numerical_columns = [
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "duration_ms"
    ]
    
    for col in numerical_columns:
        assert not sample1[col].isna().any()
        assert not sample2[col].isna().any()

def test_kolmogorov_smirnov(data):

    sample1, sample2 = data

    numerical_columns = [
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "duration_ms"
    ]

    # Let's decide the Type I error probability (related to the False Positive Rate)
    alpha = 0.05
    
    # NOTE: alpha_prime correction is due to the fact that we are apllying a test to multiple columns
    # Bonferroni correction for multiple hypothesis testing
    # (see my blog post on this topic to see where this comes from:
    # https://towardsdatascience.com/precision-and-recall-trade-off-and-multiple-hypothesis-testing-family-wise-error-rate-vs-false-71a85057ca2b)
    alpha_prime = 1 - (1 - alpha)**(1 / len(numerical_columns))
    # print(f"alpha_prime {alpha_prime}")
    
    for col in numerical_columns:

        # Use the 2-sample KS test (scipy.stats.ks_2sample) on the column
        # col
        ts, p_value = scipy.stats.ks_2samp(sample1[col], sample2[col], 
                                           alternative='two-sided')
        # print(f"{col}: p_value {p_value}")
        # Add an assertion so that the test fails if p_value > alpha_prime
        assert p_value > alpha_prime
