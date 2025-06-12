    import seaborn as sns
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Load the dataset
    transactions = pd.read_csv(r"C:\Users\becas\OneDrive\Documents\transactions_modified_credit-card-fraud-prediction.csv")

    # Preview the first 5 rows
    print("First five rows of the dataset:")
    print(transactions.head())

    # Get info about the dataset
    print("\nDataset info:")
    print(transactions.info())

