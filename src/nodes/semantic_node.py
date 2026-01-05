#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
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

class SemanticSystemNode(Node):
    def __init__(self):
        super().__init__('semantic_system_node')
        
        self.bridge = CvBridge()
        self.intrinsics = None
        self.frame_count = 0
        
        # 1. Initialize AI Modules
        self.detector = OpenVocabularyDetector(confidence=0.15) 
        self.mapper = SemanticMapper()
        
        # 2. Define Classes
        self.detector.set_classes([
            "chair", "table", "laptop", "bottle", "person", "door", "backpack", "monitor"
        ])

        # 3. TF Setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.target_frame = 'odom'
        self.camera_frame = 'camera_color_optical_frame'

        # 4. Subscribers
        self.rgb_sub = message_filters.Subscriber(
            self, Image, '/camera/color/image_raw', qos_profile=qos_profile_sensor_data)
        self.depth_sub = message_filters.Subscriber(
            self, Image, '/camera/depth/image_raw', qos_profile=qos_profile_sensor_data)
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=20, slop=0.5)
        self.ts.registerCallback(self.callback)

        self.create_subscription(CameraInfo, '/camera/color/camera_info', self.info_callback, 10)
        
        self.get_logger().info("Semantic System Ready. Waiting for video...")

    def info_callback(self, msg):
        if self.intrinsics is None:
            self.intrinsics = {'fx': msg.k[0], 'fy': msg.k[4], 'cx': msg.k[2], 'cy': msg.k[5]}
            self.camera_frame = msg.header.frame_id
            self.get_logger().info(f"Intrinsics Received. Camera Frame: {self.camera_frame}")

    def get_pose(self, time):
        try:
            # Try exact timestamp match
            t = self.tf_buffer.lookup_transform(
                self.target_frame, self.camera_frame, time, timeout=Duration(seconds=0.1))
            return self._t_to_mat(t)
        except Exception:
            # Fallback to latest available transform
            try:
                t = self.tf_buffer.lookup_transform(
                    self.target_frame, self.camera_frame, rclpy.time.Time())
                return self._t_to_mat(t)
            except:
                return None

    def _t_to_mat(self, t):
        q = [t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w]
        trans = [t.transform.translation.x, t.transform.translation.y, t.transform.translation.z]
        x, y, z, w = q
        R = np.array([
            [1 - 2*y*y - 2*z*z,  2*x*y - 2*z*w,      2*x*z + 2*y*w],
            [2*x*y + 2*z*w,      1 - 2*x*x - 2*z*z,  2*y*z - 2*x*w],
            [2*x*z - 2*y*w,      2*y*z + 2*x*w,      1 - 2*x*x - 2*y*y]
        ])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = trans
        return T

    def callback(self, rgb_msg, depth_msg):
        if self.intrinsics is None: return

        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
            depth = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
        except Exception: return

        pose = self.get_pose(rclpy.time.Time.from_msg(rgb_msg.header.stamp))
        
        if pose is None:
            cv2.putText(rgb, "WAITING FOR ODOM...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        else:
            detections = self.detector.detect_objects(rgb)
            for det in detections:
                local_pos = self.mapper.project_to_3d(det, depth, self.intrinsics)
                if local_pos is not None:
                    # Transform to global
                    p_h = np.append(local_pos, 1.0)
                    global_pos = (pose @ p_h)[:3]
                    
                    obj_id = self.mapper.update_map(det['label'], global_pos)
                    
                    # Vis
                    x, y, w, h = det['bbox']
                    cv2.rectangle(rgb, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
                    cv2.putText(rgb, f"{det['label']} {obj_id}", (int(x), int(y)-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Semantic View", rgb)
        cv2.waitKey(1)

    def save_map(self):
        path = "results/semantic_maps/latest_map.pkl"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.mapper.objects, f)
        self.get_logger().info(f"Map Saved: {path} ({len(self.mapper.objects)} objects)")

def main(args=None):
    rclpy.init(args=args)
    node = SemanticSystemNode()
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