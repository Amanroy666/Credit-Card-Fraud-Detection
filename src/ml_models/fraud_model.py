"""Fraud detection ML model training"""
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import pickle

class FraudDetectionModel:
    def __init__(self):
        self.models = {
            'rf': RandomForestClassifier(n_estimators=100),
            'gb': GradientBoostingClassifier(n_estimators=100)
        }
    
    def train(self, X, y):
        """Train ensemble of models"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            print(f"{name} accuracy: {score:.4f}")
    
    def predict_proba(self, X):
        """Ensemble prediction"""
        predictions = [model.predict_proba(X)[:, 1] for model in self.models.values()]
        return sum(predictions) / len(predictions)
    
    def save(self, filepath):
        """Save trained models"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.models, f)
