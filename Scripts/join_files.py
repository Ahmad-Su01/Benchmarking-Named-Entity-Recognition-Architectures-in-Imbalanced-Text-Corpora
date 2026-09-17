import pandas as pd
from pathlib import Path


def join_enron_csv(part1_path, part2_path, output_path):
    """
    Join two split Enron CSV files back into one CSV.

    The original row order is preserved.
    """

    part1_path = Path(part1_path)
    part2_path = Path(part2_path)
    output_path = Path(output_path)

    # Read both parts
    part1 = pd.read_csv(part1_path)
    part2 = pd.read_csv(part2_path)

    # Join them in the original order
    combined = pd.concat([part1, part2], ignore_index=True)

    # Save the reconstructed CSV
    combined.to_csv(output_path, index=False)

    print(f"Part 1 rows:  {len(part1):,}")
    print(f"Part 2 rows:  {len(part2):,}")
    print(f"Total rows:   {len(combined):,}")
    print(f"Created:      {output_path}")


join_enron_csv(
    r"..\Datasets\Cleaned\Enron\Enron_Cleaned_part1.csv",
    r"..\Datasets\Cleaned\Enron\Enron_Cleaned_part2.csv",
    r"..\Datasets\Cleaned\Enron\Enron_Cleaned.csv"
)

join_enron_csv(
    r"..\Datasets\NER\Enron\Enron_NER_part1.csv",
    r"..\Datasets\NER\Enron\Enron_NER_part2.csv",
    r"..\Datasets\NER\Enron\Enron_NER.csv"
)

join_enron_csv(
    r"..\Datasets\Baseline\Enron\Enron_Baseline_part1.csv",
    r"..\Datasets\Baseline\Enron\Enron_Baseline_part2.csv",
    r"..\Datasets\Baseline\Enron\Enron_Baseline.csv"
)