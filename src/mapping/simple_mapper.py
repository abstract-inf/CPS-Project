import numpy as np

class SemanticMapper:
    def __init__(self):
        self.objects = {} 
        self.obj_counter = 0
        self.merge_threshold = 0.5 

    def project_to_3d(self, bbox, depth_img, intrinsics):
        # FIX: Directly unpack the list/tuple [x, y, w, h]
        x, y, w, h = bbox
        
        # Clamp to image dimensions
        h_img, w_img = depth_img.shape
        x1, y1 = max(0, int(x)), max(0, int(y))
        x2, y2 = min(w_img, int(x+w)), min(h_img, int(y+h))

        roi = depth_img[y1:y2, x1:x2]
        valid_depths = roi[roi > 0]
        
        if len(valid_depths) == 0: return None

        z_mm = np.median(valid_depths)
        z = z_mm / 1000.0 

        if z <= 0.1 or z > 8.0: return None

        cx_img = x + w/2
        cy_img = y + h/2
        
        fx, fy = intrinsics['fx'], intrinsics['fy']
        cx, cy = intrinsics['cx'], intrinsics['cy']

        X = (cx_img - cx) * z / fx
        Y = (cy_img - cy) * z / fy

        return np.array([X, Y, z])

    def transform_to_global(self, point_3d, camera_pose):
        p_h = np.append(point_3d, 1.0)
        p_global = camera_pose @ p_h
        return p_global[:3]

    def update_map(self, label, global_pos):
        best_id = -1
        min_dist = float('inf')

        for obj_id, data in self.objects.items():
            if data['label'] == label:
                dist = np.linalg.norm(data['centroid'] - global_pos)
                if dist < min_dist:
                    min_dist = dist
                    best_id = obj_id

        if min_dist < self.merge_threshold:
            obj = self.objects[best_id]
            n = obj['count']
            obj['centroid'] = (obj['centroid'] * n + global_pos) / (n + 1)
            obj['count'] += 1
            return best_id
        else:
            new_id = self.obj_counter
            self.objects[new_id] = {
                'label': label,
                'centroid': global_pos,
                'count': 1
            }
            self.obj_counter += 1
            return new_id