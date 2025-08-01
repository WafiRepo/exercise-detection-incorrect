import pickle
import json
import os

def generate_labels_from_pkl(pkl_file_path, output_json_path):
    """
    Generate labels JSON file from a pickle model file.
    This assumes the model has a classes_ attribute or similar.
    """
    try:
        # Load the pickle model
        with open(pkl_file_path, 'rb') as f:
            model = pickle.load(f)
        
        # Try different common attribute names for classes
        classes = None
        if hasattr(model, 'classes_'):
            classes = model.classes_
        elif hasattr(model, 'classes'):
            classes = model.classes
        elif hasattr(model, 'class_names'):
            classes = model.class_names
        elif hasattr(model, 'labels'):
            classes = model.labels
        
        if classes is not None:
            # Create labels dictionary
            labels_dict = {str(i): str(label) for i, label in enumerate(classes)}
            
            # Save to JSON
            with open(output_json_path, 'w') as f:
                json.dump(labels_dict, f, indent=2)
            
            print(f"Labels generated successfully!")
            print(f"Found {len(classes)} classes:")
            for i, label in enumerate(classes):
                print(f"  {i}: {label}")
            print(f"Saved to: {output_json_path}")
            
        else:
            print("Could not find classes in the model.")
            print("Available attributes:", dir(model))
            print("\nCreating default labels file...")
            
            # Create default labels based on common squat errors
            default_labels = {
                "0": "correct",
                "1": "knees_too_forward", 
                "2": "back_too_rounded",
                "3": "knees_caving_in",
                "4": "heels_lifting",
                "5": "depth_insufficient"
            }
            
            with open(output_json_path, 'w') as f:
                json.dump(default_labels, f, indent=2)
            
            print(f"Default labels saved to: {output_json_path}")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Creating default labels file...")
        
        # Create default labels
        default_labels = {
            "0": "correct",
            "1": "knees_too_forward",
            "2": "back_too_rounded", 
            "3": "knees_caving_in",
            "4": "heels_lifting",
            "5": "depth_insufficient"
        }
        
        with open(output_json_path, 'w') as f:
            json.dump(default_labels, f, indent=2)
        
        print(f"Default labels saved to: {output_json_path}")

if __name__ == "__main__":
    # Update these paths to match your actual file locations
    pkl_file = "E:/Holowellness/algorithm\Code\models\squat\squat_2.pkl"  # Update this path
    output_file = "e:/Holowellness/algorithm/Code/models/squat/squat_2_labels.json"  # Update this path
    
    if os.path.exists(pkl_file):
        generate_labels_from_pkl(pkl_file, output_file)
    else:
        print(f"PKL file not found at: {pkl_file}")
        print("Please update the path in this script to point to your squat_detection.pkl file") 