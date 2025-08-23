
if __name__ == "__main__":
    columns = [
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
    ks_alphas = [0.05, 0.95, 0.97, 0.99]
    
    for ks_alpha in ks_alphas:
        alpha_prime = 1 - (1 - ks_alpha)**(1 / len(columns))
        print(f"ks_alpha: {ks_alpha}, alpha_prime: {alpha_prime}")