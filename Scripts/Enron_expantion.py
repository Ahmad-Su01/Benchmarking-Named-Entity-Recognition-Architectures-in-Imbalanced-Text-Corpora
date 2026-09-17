import os
import pandas as pd

# Root directory containing enron1 ... enron6
DATA_DIR = r"G:/My Drive/Dissertation/Datasets/Baseline/Enron"

all_data = []

# Loop through enron1 to enron6
for i in range(1, 7):
    dataset_name = f"enron{i}"
    dataset_path = os.path.join(DATA_DIR, dataset_name)

    if not os.path.exists(dataset_path):
        print(f"Folder not found: {dataset_path}")
        continue

    print(f"Processing {dataset_name}...")

    # Process ham and spam folders
    for label in ["ham", "spam"]:
        label_path = os.path.join(dataset_path, label)

        if not os.path.exists(label_path):
            print(f"Missing folder: {label_path}")
            continue

        for root, dirs, files in os.walk(label_path):

            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            count   = 0

            print(f"{dataset_name} - {label}: ")

            for file in files:
                
                count += 1

                # Skip hidden files
                if file.startswith("."):
                    continue

                # Optional: only process text files
                if not file.lower().endswith(".txt"):
                    continue

                file_path = os.path.join(root, file)

                try:
                    with open(file_path, "r", encoding="latin-1") as f:
                        content = f.read()

		
                    all_data.append({
                        "text": content,
                        "label": label,
                        "dataset": dataset_name,
                        "filename": file
                    })

                except Exception as e:
                    print(f"Skipping {file_path}: {e}")

            print(count)

# Create DataFrame
df = pd.DataFrame(all_data)

# Save to Enron CSV
output_file = r"G:/My Drive/Dissertation/Datasets/Baseline/Enron.csv"
df.to_csv(output_file, index=False)

# Summary
print("\nDataset Summary")
print("-" * 30)
print(df["label"].value_counts())
print(f"\nTotal emails: {len(df):,}")
print(f"Saved to: {output_file}")