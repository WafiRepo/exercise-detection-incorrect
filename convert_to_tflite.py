import pickle
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

def convert_sklearn_to_tflite(model_path, output_path):
    """
    Convert scikit-learn model to TensorFlow Lite format
    """
    print(f"Loading model from: {model_path}")
    
    # Load the scikit-learn model
    with open(model_path, 'rb') as f:
        sklearn_model = pickle.load(f)
    
    print(f"Model type: {type(sklearn_model)}")
    print(f"Model classes: {sklearn_model.classes_}")
    
    # Create a simple TensorFlow model that mimics the sklearn model
    def create_tf_model(input_dim, num_classes):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        return model
    
    # Determine input dimensions (MediaPipe pose has 33 landmarks * 4 features = 132)
    input_dim = 132  # x, y, z, visibility for each landmark
    num_classes = len(sklearn_model.classes_)
    
    print(f"Creating TensorFlow model with input_dim={input_dim}, num_classes={num_classes}")
    
    # Create TensorFlow model
    tf_model = create_tf_model(input_dim, num_classes)
    
    # Generate synthetic training data to train the TF model
    print("Generating synthetic training data...")
    num_samples = 1000
    X_synthetic = np.random.randn(num_samples, input_dim)
    
    # Use sklearn model to generate labels
    y_synthetic = sklearn_model.predict(X_synthetic)
    
    # Convert labels to one-hot encoding
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_synthetic)
    y_one_hot = tf.keras.utils.to_categorical(y_encoded, num_classes)
    
    # Train the TensorFlow model
    print("Training TensorFlow model...")
    tf_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train for a few epochs
    tf_model.fit(
        X_synthetic, y_one_hot,
        epochs=10,
        batch_size=32,
        verbose=1
    )
    
    # Convert to TensorFlow Lite
    print("Converting to TensorFlow Lite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
    
    # Optimize for size and speed
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Convert
    tflite_model = converter.convert()
    
    # Save the model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TensorFlow Lite model saved to: {output_path}")
    print(f"Model size: {len(tflite_model) / 1024:.2f} KB")
    
    # Save label mapping
    label_mapping = {i: label for i, label in enumerate(sklearn_model.classes_)}
    label_path = output_path.replace('.tflite', '_labels.json')
    
    import json
    with open(label_path, 'w') as f:
        json.dump(label_mapping, f, indent=2)
    
    print(f"Label mapping saved to: {label_path}")
    
    return output_path, label_path

def create_flutter_integration_code(model_path, label_path):
    """
    Create Flutter integration code
    """
    flutter_code = '''
// Flutter integration code for TensorFlow Lite model
import 'package:tflite_flutter/tflite_flutter.dart';
import 'dart:convert';
import 'dart:io';

class SquatClassifier {
  late Interpreter _interpreter;
  late Map<int, String> _labelMap;
  
  Future<void> loadModel() async {
    // Load TensorFlow Lite model
    _interpreter = await Interpreter.fromAsset('assets/squat_model.tflite');
    
    // Load label mapping
    final labelFile = await File('assets/squat_model_labels.json').readAsString();
    final labelData = json.decode(labelFile);
    _labelMap = Map<int, String>.from(labelData);
    
    print('Model loaded successfully');
    print('Labels: $_labelMap');
  }
  
  Future<String> predict(List<double> features) async {
    if (features.length != 132) {
      throw Exception('Expected 132 features, got ${features.length}');
    }
    
    // Prepare input
    var input = [features];
    
    // Prepare output
    var output = List.filled(1 * _labelMap.length, 0.0).reshape([1, _labelMap.length]);
    
    // Run inference
    _interpreter.run(input, output);
    
    // Get prediction
    var predictions = output[0];
    var maxIndex = predictions.indexOf(predictions.reduce((a, b) => a > b ? a : b));
    
    return _labelMap[maxIndex] ?? 'unknown';
  }
  
  Future<String> predictFromLandmarks(List<Map<String, double>> landmarks) async {
    // Convert MediaPipe landmarks to feature vector
    var features = <double>[];
    
    for (var landmark in landmarks) {
      features.add(landmark['x'] ?? 0.0);
      features.add(landmark['y'] ?? 0.0);
      features.add(landmark['z'] ?? 0.0);
      features.add(landmark['visibility'] ?? 0.0);
    }
    
    return await predict(features);
  }
  
  void dispose() {
    _interpreter.close();
  }
}

// Usage example:
// final classifier = SquatClassifier();
// await classifier.loadModel();
// String result = await classifier.predictFromLandmarks(landmarks);
// print('Prediction: $result');
'''
    
    with open('flutter_integration.dart', 'w') as f:
        f.write(flutter_code)
    
    print("Flutter integration code saved to: flutter_integration.dart")

def main():
    # Model paths
    model_path = "./models/squat/squat_2.pkl"
    output_path = "./models/squat/squat_2.tflite"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    try:
        # Convert model
        tflite_path, label_path = convert_sklearn_to_tflite(model_path, output_path)
        
        # Create Flutter integration code
        create_flutter_integration_code(tflite_path, label_path)
        
        print("\\n" + "="*50)
        print("CONVERSION COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"TensorFlow Lite model: {tflite_path}")
        print(f"Label mapping: {label_path}")
        print("Flutter integration code: flutter_integration.dart")
        print("\\nNext steps:")
        print("1. Copy squat_model.tflite to your Flutter assets folder")
        print("2. Copy squat_model_labels.json to your Flutter assets folder")
        print("3. Use the provided Flutter integration code")
        print("4. Add tflite_flutter dependency to pubspec.yaml")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 