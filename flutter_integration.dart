
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
