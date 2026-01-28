#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from cv_bridge import CvBridge
import message_filters
import numpy as np
import cv2
import sys
import os
import pickle
from tf2_ros import Buffer, TransformListener

# Add 'src' to python path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from perception.yolo_detector import OpenVocabularyDetector
from mapping.simple_mapper import SemanticMapper

class InteractiveSemanticNode(Node):
    def __init__(self):
        super().__init__('semantic_system_node')
        
        self.bridge = CvBridge()
        self.intrinsics = None
        self.latest_pose = None
        
        # --- Navigation State ---
        self.target_label = None
        self.nav_state = "IDLE"
        self.target_id = None
        
        # --- Modules ---
        self.detector = OpenVocabularyDetector(confidence=0.15) 
        self.mapper = SemanticMapper()
        
        self.class_list = [
            "chair", "table", "laptop", "bottle", "person", "door", 
            "backpack", "monitor", "keyboard", "mouse"
        ]
        self.detector.set_classes(self.class_list)

        # --- TF Setup ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.fixed_frame = 'odom'
        self.camera_frame = 'camera_color_optical_frame'

        # --- Subscribers ---
        self.rgb_sub = message_filters.Subscriber(
            self, Image, '/camera/color/image_raw', qos_profile=qos_profile_sensor_data)
        self.depth_sub = message_filters.Subscriber(
            self, Image, '/camera/depth/image_raw', qos_profile=qos_profile_sensor_data)
        
        # NEW: Listen for commands from separate script
        self.cmd_sub = self.create_subscription(String, '/navigation_command', self.command_callback, 10)
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=20, slop=0.5)
        self.ts.registerCallback(self.callback)

        self.create_subscription(CameraInfo, '/camera/color/camera_info', self.info_callback, 10)
        
        self.get_logger().info("System Ready. Waiting for commands on /navigation_command topic...")

    def command_callback(self, msg):
        cmd = msg.data.lower()
        self.get_logger().info(f"Received Command: {cmd}")
        
        if cmd == "stop":
            self.nav_state = "IDLE"
            self.target_label = None
        elif any(x in cmd for x in ["navigate", "move", "find"]):
            # Simple parsing: "find chair" -> "chair"
            target = cmd.split()[-1]
            if target in self.class_list:
                self.target_label = target
                self.nav_state = "SEARCHING"
                self.target_id = None
            else:
                self.get_logger().warn(f"Unknown object: {target}")

    def info_callback(self, msg):
        if self.intrinsics is None:
            self.intrinsics = {'fx': msg.k[0], 'fy': msg.k[4], 'cx': msg.k[2], 'cy': msg.k[5]}

    def get_pose(self, time):
        try:
            t = self.tf_buffer.lookup_transform(
                self.fixed_frame, self.camera_frame, time, timeout=Duration(seconds=0.1))
            return self._t_to_mat(t)
        except:
            try:
                t = self.tf_buffer.lookup_transform(
                    self.fixed_frame, self.camera_frame, rclpy.time.Time())
                return self._t_to_mat(t)
            except:
                return None

    def _t_to_mat(self, t):
        q = [t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w]
        tr = [t.transform.translation.x, t.transform.translation.y, t.transform.translation.z]
        x,y,z,w = q
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tr
        return T

    def project_point_to_pixel(self, point_3d):
        if self.latest_pose is None: return None
        R = self.latest_pose[:3, :3]
        t = self.latest_pose[:3, 3]
        point_cam = R.T @ (point_3d - t)
        
        if point_cam[2] <= 0: return None
        
        fx, fy = self.intrinsics['fx'], self.intrinsics['fy']
        cx, cy = self.intrinsics['cx'], self.intrinsics['cy']
        
        u = int(point_cam[0] * fx / point_cam[2] + cx)
        v = int(point_cam[1] * fy / point_cam[2] + cy)
        return (u, v)

    def callback(self, rgb_msg, depth_msg):
        if self.intrinsics is None: return
        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
            depth = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
        except: return

        pose = self.get_pose(rclpy.time.Time.from_msg(rgb_msg.header.stamp))
        self.latest_pose = pose

        if pose is None:
            cv2.putText(rgb, "WAITING FOR ODOM...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        else:
            detections = self.detector.detect_objects(rgb)
            for det in detections:
                # Pass bbox list directly
                local_pos = self.mapper.project_to_3d(det['bbox'], depth, self.intrinsics)
                if local_pos is not None:
                    p_h = np.append(local_pos, 1.0)
                    global_pos = (pose @ p_h)[:3]
                    obj_id = self.mapper.update_map(det['label'], global_pos)
                    
                    x, y, w, h = det['bbox']
                    cv2.rectangle(rgb, (int(x), int(y)), (int(x+w), int(y+h)), (0,255,0), 2)
                    cv2.putText(rgb, f"{det['label']} {obj_id}", (int(x), int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            self.draw_navigation_overlay(rgb, pose)

        cv2.imshow("Interactive Navigation", rgb)
        cv2.waitKey(1)

    def draw_navigation_overlay(self, img, pose):
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        current_pos = pose[:3, 3]
        
        overlay = [f"State: {self.nav_state}"]
        
        if self.nav_state != "IDLE":
            overlay.append(f"Target: {self.target_label}")
            candidates = []
            closest_any = None
            min_dist_any = float('inf')
            
            for oid, obj in self.mapper.objects.items():
                d = np.linalg.norm(obj['centroid'] - current_pos)
                if d < min_dist_any:
                    min_dist_any = d
                    closest_any = obj
                if self.target_label in obj['label'].lower():
                    candidates.append((d, obj, oid))
            
            if candidates:
                candidates.sort(key=lambda x: x[0])
                d, target, oid = candidates[0]
                overlay.append(f"Found ID: {oid}")
                overlay.append(f"Dist: {d:.2f}m")
                overlay.append(f"Loc: [{target['centroid'][0]:.1f}, {target['centroid'][1]:.1f}, {target['centroid'][2]:.1f}]")
                
                px = self.project_point_to_pixel(target['centroid'])
                if px:
                    cv2.arrowedLine(img, center, px, (0, 255, 0), 4)
            else:
                overlay.append("Status: Searching...")
                if closest_any:
                    overlay.append(f"Hint: Near {closest_any['label']}?")
                    px = self.project_point_to_pixel(closest_any['centroid'])
                    if px:
                        cv2.arrowedLine(img, center, px, (255, 0, 0), 3)

        cv2.rectangle(img, (10, 10), (350, 20 + len(overlay)*30), (0,0,0), -1)
        for i, txt in enumerate(overlay):
            cv2.putText(img, txt, (20, 40 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)

    def save_map(self):
        path = "results/semantic_maps/latest_map.pkl"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.mapper.objects, f)
        self.get_logger().info("Map Saved.")

def main(args=None):
    rclpy.init(args=args)
    node = InteractiveSemanticNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_map()
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()