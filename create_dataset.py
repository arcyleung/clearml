import datasets
from clearml import StorageManager

ds = datasets.load_from_disk("/home/arthur/Projects/arkts/arks_code_dataset_v1")
ds.to_parquet("/home/arthur/Projects/arkts/arks_code_dataset_v1_parquet")

manager = StorageManager()
dataset_path = manager.get_local_copy(
    remote_url="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
)
# make sure to copy the printed value
print("COPY THIS DATASET PATH: {}".format(dataset_path))

# clearml-data create --project dataset_examples --name cifar_dataset

# print(ds[0]["content"])


