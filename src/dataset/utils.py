import pandas as pd

def print_df_info(df: pd.DataFrame, name: str, verbose: bool = False) -> None:
    if not verbose:
        return

    print(f"{name}:")
    print(f"Shape: {df.shape}")
    print(df.head().to_string())
    print()
