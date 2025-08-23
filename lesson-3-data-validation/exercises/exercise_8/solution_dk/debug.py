import pandas as pd 


if __name__ == '__main__':
    
    sample1 = pd.read_csv('sample1.csv')
    sample2 = pd.read_csv('sample2.csv')
    
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

    print("sample1")
    for col in numerical_columns:
        # Check if a specific column contains NaN
        contains_nan = sample1[col].isna().any()
        if contains_nan:
            print(f"Column {col} contains NaN: {contains_nan}")
            print(sample1[col].sum())
        
    print("sample2")
    for col in numerical_columns:
        # Check if a specific column contains NaN
        contains_nan = sample2[col].isna().any()
        if contains_nan:
            print(f"Column {col} contains NaN: {contains_nan}")
            print(sample2[col].sum())
            