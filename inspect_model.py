import pickle

# Load your trained model
model = pickle.load(open("ckd_model.pkl", "rb"))

# Print the feature names used when the model was trained
print("Model expects the following feature names:\n")
print(list(model.feature_names_in_))
