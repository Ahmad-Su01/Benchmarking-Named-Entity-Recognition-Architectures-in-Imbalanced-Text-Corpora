# import os
# import tarfile
# import pandas as pd

# # Path where your tar files are located
# DATA_DIR = ""  # change if needed

# # Mapping file → label
# datasets = {
#     "20030228_easy_ham.tar.bz2": "0",
#     "20030228_easy_ham_2.tar.bz2": "0",
#     "20030228_hard_ham.tar.bz2": "0",
#     "20030228_spam.tar.bz2": "1",
#     "20030228_spam_2.tar.bz2": "1",
# }

# all_data = []

# for tar_name, label in datasets.items():
#     tar_path = os.path.join(DATA_DIR, tar_name)
#     extract_path = os.path.join(DATA_DIR, tar_name.replace(".tar", ""))

#     # Extract if not already extracted
#     if not os.path.exists(extract_path):
#         with tarfile.open(tar_path, "r") as tar:
#             tar.extractall(path=extract_path)

#     # Read files
#     for root, _, files in os.walk(extract_path):
#         for file in files:
#             file_path = os.path.join(root, file)

#             try:
#                 with open(file_path, "r", encoding="latin-1") as f:
#                     content = f.read()
#                     all_data.append({
#                         "text": content,
#                         "label": label
#                     })
#             except Exception as e:
#                 print(f"Skipping {file_path}: {e}")

# # Create DataFrame
# df = pd.DataFrame(all_data)
# df.to_csv("SpamAssassin_ChatGPT.csv", index=False)
# print(df.head())
# print(df["label"].value_counts())
# print("Total emails: ", len(df))

import os
import tarfile
import pandas as pd

DATA_DIR = ""  # change if needed

# datasets_tar = {
#     "20030228_easy_ham.tar.bz2": "0",
#     "20030228_easy_ham_2.tar.bz2": "0",
#     "20030228_hard_ham.tar.bz2": "0",
#     "20030228_spam.tar.bz2": "1",
#     "20030228_spam_2.tar.bz2": "1",
# }

datasets = {
    "easy_ham": "0",
    "easy_ham_2": "0",
    "hard_ham": "0",
    "spam": "1",
    "spam_2": "1",
}

# Files/directories to skip (non-email metadata in SpamAssassin corpus)
SKIP_NAMES = {"cmds", "index", "README", "LICENSE", ".DS_Store"}

all_data = []

for dataset, label in datasets.items(): # tar_name, label -> dataset, label
    # tar_path = os.path.join(DATA_DIR, tar_name)
    # extract_path = os.path.join(DATA_DIR, tar_name.replace(".tar", ""))
    
    extract_path = os.path.join(DATA_DIR, dataset)
    
    # Extract if not already extracted
    if not os.path.exists(extract_path):
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=extract_path)

    # Walk the extracted directory
    for root, dirs, files in os.walk(extract_path):
        # Skip hidden directories (like __MACOSX)
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in SKIP_NAMES]
        
        for file in files:
            # Skip hidden files, metadata files, and non-mail files
            if file.startswith(".") or file in SKIP_NAMES:
                continue
            
            file_path = os.path.join(root, file)

            try:
                with open(file_path, "r", encoding="latin-1") as f:
                    content = f.read()
                    
                    # Optional: skip if content looks like a command list, not an email
                    # (SpamAssassin 'cmds' files list shell commands)
                    if content.strip().startswith("#!") or content.strip().startswith("cd "):
                        continue
                    
                    all_data.append({
                        "text": content,
                        "label": label
                    })
            except Exception as e:
                print(f"Skipping {file_path}: {e}")

df = pd.DataFrame(all_data)
df.to_csv("G:/My Drive/Dissertation/Datasets/Baseline/SpamAssassin.csv", index=False)
print(df["label"].value_counts())
print("Total emails:", len(df))