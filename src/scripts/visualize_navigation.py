import pickle
import matplotlib.pyplot as plt
import os
import sys

def main():
    path = "results/semantic_maps/latest_map.pkl"
    if not os.path.exists(path):
        print("No map found! Run the system first.")
        return

    with open(path, 'rb') as f:
        data = pickle.load(f)

    print(f"Loaded map with {len(data)} objects.")

    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Extract coordinates
    x_vals, y_vals, labels = [], [], []
    
    for obj_id, obj in data.items():
        # obj['centroid'] is [x, y, z] in global frame
        # We plot X and Z (Top-down view usually uses X-Z or X-Y depending on camera orientation)
        # Standard ROS camera: Z is forward, X is right, Y is down.
        # So Top-Down map is X vs Z.
        cx = obj['centroid'][0]
        cz = obj['centroid'][2] 
        
        x_vals.append(cx)
        y_vals.append(cz)
        labels.append(f"{obj['label']} {obj_id}")

    ax.scatter(x_vals, y_vals, c='blue', marker='o')
    
    for i, txt in enumerate(labels):
        ax.annotate(txt, (x_vals[i], y_vals[i]), fontsize=9)

    ax.set_xlabel("X (Right/Left) [meters]")
    ax.set_ylabel("Z (Forward/Back) [meters]")
    ax.set_title("Semantic Map Top-Down View")
    ax.grid(True)
    ax.axis('equal')
    
    plt.show()

if __name__ == "__main__":
    main()