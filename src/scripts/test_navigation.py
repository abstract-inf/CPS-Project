#!/usr/bin/env python3
import pickle
import numpy as np
import os
import sys

def main():
    # 1. Load the Map
    # Ensure this path matches where semantic_node.py saves the file
    map_path = "results/semantic_maps/latest_map.pkl"
    
    if not os.path.exists(map_path):
        print(f"Error: Map file not found at {map_path}")
        print("Run the mapping node first and press Ctrl+C to save the map.")
        return

    print(f"Loading map from {map_path}...")
    with open(map_path, 'rb') as f:
        semantic_map = pickle.load(f)

    print(f"Map loaded with {len(semantic_map)} objects.")
    
    # 2. Define Test Queries
    # These mimic what a user might type
    queries = [
        "chair",
        "bottle",
        "laptop",
        "person",
        "table"
    ]

    # 3. Simulated Navigation
    # We assume the robot starts at (0,0,0) in the Odom frame
    current_pos = np.array([0.0, 0.0, 0.0])
    
    print("\n--- Starting Navigation Tests ---\n")

    for query in queries:
        print(f"Query: 'Find a {query}'")
        
        # Simple Logic: Find closest object matching label
        best_target = None
        min_dist = float('inf')

        for obj_id, data in semantic_map.items():
            # Check if label matches query (simple string match for YOLO)
            if query.lower() in data['label'].lower():
                # Calculate distance
                dist = np.linalg.norm(data['centroid'] - current_pos)
                if dist < min_dist:
                    min_dist = dist
                    best_target = data
                    best_target['id'] = obj_id

        if best_target:
            target_pos = best_target['centroid']
            print(f"  -> Found object ID {best_target['id']} ({best_target['label']})")
            print(f"  -> Location: [X={target_pos[0]:.2f}, Y={target_pos[1]:.2f}, Z={target_pos[2]:.2f}]")
            print(f"  -> Distance: {min_dist:.2f} meters")
            print("  -> Action: Navigating to target...")
            # In a real robot, you would send target_pos to Nav2 here
        else:
            print(f"  -> Object '{query}' not found in map.")
        
        print("-" * 30)

if __name__ == "__main__":
    main()