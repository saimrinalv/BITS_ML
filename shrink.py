import joblib

# Load your large 30MB model
model = joblib.load('Random Forest.pkl')

# Re-save it with compression (this will shrink it to around 3MB - 5MB!)
joblib.dump(model, 'Random Forest.pkl', compress=3)

print("Model compressed successfully!")