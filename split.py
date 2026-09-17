import pandas as pd
from pathlib import Path


def split_enron_csv(file_path):
    file_path = Path(file_path)

    df = pd.read_csv(file_path)

    midpoint = len(df) // 2

    part1 = df.iloc[:midpoint]
    part2 = df.iloc[midpoint:]

    part1_path = file_path.with_name(
        f"{file_path.stem}_part1{file_path.suffix}"
    )

    part2_path = file_path.with_name(
        f"{file_path.stem}_part2{file_path.suffix}"
    )

    part1.to_csv(part1_path, index=False)
    part2.to_csv(part2_path, index=False)

    print(f"Original rows: {len(df):,}")
    print(f"Part 1 rows:   {len(part1):,}")
    print(f"Part 2 rows:   {len(part2):,}")
    print(f"Created: {part1_path}")
    print(f"Created: {part2_path}")


# Change these paths if necessary
split_enron_csv(r"Datasets\Cleaned\Enron\Enron_Cleaned.csv")
split_enron_csv(r"Datasets\NER\Enron\Enron_NER.csv")
split_enron_csv(r"Datasets\Baseline\Enron\Enron_Baseline.csv")