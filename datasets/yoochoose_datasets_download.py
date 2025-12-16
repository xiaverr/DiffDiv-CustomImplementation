import kagglehub

# Download latest version
path = kagglehub.dataset_download("chadgostopp/recsys-challenge-2015")

print("Path to dataset files:", path)