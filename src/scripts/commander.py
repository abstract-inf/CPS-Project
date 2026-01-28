#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import sys

class Commander(Node):
    def __init__(self):
        super().__init__('commander_node')
        self.pub = self.create_publisher(String, '/navigation_command', 10)
        print("\n" + "="*50)
        print("  COMMAND CENTER")
        print("  Type: 'navigate to <object>' or 'stop'")
        print("  Press Ctrl+C to exit.")
        print("="*50 + "\n")

    def run_loop(self):
        while rclpy.ok():
            try:
                cmd = input(">> ").strip()
                if not cmd: continue
                
                msg = String()
                msg.data = cmd
                self.pub.publish(msg)
                print(f"Sent: {cmd}")
            except EOFError:
                break
            except KeyboardInterrupt:
                break

def main():
    rclpy.init()
    node = Commander()
    try:
        node.run_loop()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()