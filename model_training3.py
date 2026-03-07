from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib # To save the model for Webots
import pandas as pd

# Load data
df = pd.read_csv('./horizon_videos/horizon_videos_3/horizon_training_data.csv')
X = df[['edge_density', 'relative_length']]
y = df['label']

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Save the 'brain'
joblib.dump(clf, 'horizon_validator.pkl')
print("Model trained and saved as horizon_validator.pkl")
