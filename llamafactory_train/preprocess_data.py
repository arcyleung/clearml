import pandas as pd

input_parquet = "dataset.parquet"     # Path to your input Parquet file
output_json = "dataset.json"          # Path to your output JSON file

# Load the parquet file into a pandas DataFrame
df = pd.read_parquet(input_parquet, engine="pyarrow")

# Convert DataFrame to JSON
# orient="records" creates a list of dictionaries for each row
# lines=True outputs in JSON Lines (one JSON object per line)
df.to_json(output_json, orient="records", lines=True, force_ascii=False)

print(f"Successfully converted {input_parquet} to {output_json}")

