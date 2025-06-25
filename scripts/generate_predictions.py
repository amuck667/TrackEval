#!/usr/bin/env python3
# created to generate synthetic prediction data from ground truth annotations
# usecase: keyponts MOT format data - frame, class, track, and keypoints with visibility

import random
import numpy as np

def modify_data_for_predictions(input_file, output_file):
    """
    Copy data from input file and modify it to simulate predictions with:
    - Class misclassifications (0-5)
    - Track swaps (0-6) 
    - Keypoint variations (10-50%)
    - Visibility as prediction confidence (unchanged)
    """
    
    # Set seed for reproducible results
    random.seed(42)
    np.random.seed(42)
    
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
                
            # Parse the line
            parts = line.split(',')
            
            # Extract basic info
            frame = int(parts[0])
            original_class = int(parts[1])
            original_track = int(parts[2])
            
            # Simulate class misclassification (10% chance)
            if random.random() < 0.1:
                new_class = random.randint(0, 5)
                while new_class == original_class:
                    new_class = random.randint(0, 5)
            else:
                new_class = original_class
            
            # Simulate track swap (15% chance)
            if random.random() < 0.15:
                new_track = random.randint(0, 6)
                while new_track == original_track:
                    new_track = random.randint(0, 6)
            else:
                new_track = original_track
            
            # Keep the -1 placeholders
            modified_parts = [str(frame), str(new_class), str(new_track), '-1', '-1', '-1', '-1']
            
            # Process keypoints (starting from index 7)
            keypoint_data = parts[7:]
            
            # Keypoints come in groups of 3: x, y, visibility
            num_keypoints = len(keypoint_data) // 3
            
            for i in range(num_keypoints):
                kp_start_idx = i * 3
                x = float(keypoint_data[kp_start_idx])
                y = float(keypoint_data[kp_start_idx + 1])
                # visibility = keypoint_data[kp_start_idx + 2]  # Keep as string (unchanged)

                # Vary keypoints by 10-50%
                variation_percent = random.uniform(0.1, 0.5)  # 10-50%
                
                # Apply random variation to x and y coordinates
                x_variation = x * variation_percent * random.uniform(-1, 1)
                y_variation = y * variation_percent * random.uniform(-1, 1)
                
                new_x = x + x_variation
                new_y = y + y_variation
                
                # Ensure coordinates don't go negative (clamp to 0)
                new_x = max(0.0, new_x)
                new_y = max(0.0, new_y)
                
                # Generate random confidence score between 0 and 1
                confidence = "{:.3f}".format(random.uniform(0, 1))

                # Format to 3 decimal places to match original format
                # modified_parts.extend(["{:.3f}".format(new_x), "{:.3f}".format(new_y), visibility])  # to keep visibility
                modified_parts.extend(["{:.3f}".format(new_x), "{:.3f}".format(new_y), confidence])

            # Write the modified line
            outfile.write(','.join(modified_parts) + '\n')

def main():
    input_file = '../data/gt/P11H.txt'
    output_file = '../data/trackers/P11H_pred.txt'
    
    print("Processing {} to create {}...".format(input_file, output_file))
    print("Applying modifications:")
    print("- Class misclassifications: 10% chance")
    print("- Track swaps: 15% chance") 
    print("- Keypoint variations: 10-50% of original values")
    print("- Visibility values kept as prediction confidence")
    
    modify_data_for_predictions(input_file, output_file)
    
    print("Successfully created {}".format(output_file))
    
    # Show some statistics
    with open(input_file, 'r') as f:
        original_lines = len(f.readlines())
    
    with open(output_file, 'r') as f:
        modified_lines = len(f.readlines())
    
    print("Original file: {} lines".format(original_lines))
    print("Modified file: {} lines".format(modified_lines))

if __name__ == "__main__":
    main()
