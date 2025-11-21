"""Real-time fraud detection using Kafka Streams"""
from kafka import KafkaConsumer, KafkaProducer
import json
import pickle

class RealtimeFraudDetector:
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        self.consumer = KafkaConsumer(
            'transactions',
            bootstrap_servers='localhost:9092',
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        
        self.producer = KafkaProducer(
            bootstrap_servers='localhost:9092',
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
    
    def detect(self):
        """Real-time fraud detection"""
        for message in self.consumer:
            transaction = message.value
            features = self.extract_features(transaction)
            fraud_probability = self.model.predict_proba([features])[0]
            
            if fraud_probability > 0.8:
                self.producer.send('fraud-alerts', {
                    'transaction_id': transaction['id'],
                    'probability': float(fraud_probability),
                    'timestamp': transaction['timestamp']
                })
    
    def extract_features(self, transaction):
        """Extract features from transaction"""
        return []  # Feature extraction logic
