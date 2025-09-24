import joblib
import sys
import pandas as pd

model = joblib.load("model.joblib")
sample = [[float(x) for x in sys.argv[1:5]]]
print("Predicted class:", model.predict(sample))
