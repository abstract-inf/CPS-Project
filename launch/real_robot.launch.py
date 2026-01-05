import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    
    # Get project root dynamically
    project_root = os.getcwd()
    
    # 1. Hardware Driver (Orbbec Femto Mega)
    orbbec_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('orbbec_camera'), 'launch', 'femto_mega.launch.py')
        ),
        launch_arguments={
            'depth_registration': 'true',  
            'enable_sync': 'true',         
            'publish_tf': 'true'           
        }.items()
    )

    # 2. Visual Odometry (RTAB-Map)
    rgbd_odometry = Node(
        package='rtabmap_odom',
        executable='rgbd_odometry',
        output='screen',
        arguments=['--delete_db_on_start'],
        remappings=[
            ('rgb/image', '/camera/color/image_raw'),
            ('depth/image', '/camera/depth/image_raw'),
            ('rgb/camera_info', '/camera/color/camera_info'),
            ('odom', '/odom')
        ],
        parameters=[{
            'frame_id': 'camera_link',
            'odom_frame_id': 'odom',
            'publish_tf': True,
            'wait_imu_to_init': False,
            'approx_sync': True,          # Set True to handle "Time difference high"
            'queue_size': 20,             # Buffer for sync
            'Vis/MinInliers': '10',       # Lower requirement for tracking
            'Odom/Strategy': '1',         # Frame-to-Frame (faster)
            'Odom/ResetCountdown': '15'   # Auto-reset if lost
        }]
    )

    # 3. Static TF (Base -> Camera)
    # Required for visualization context
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'camera_link']
    )

    # 4. Semantic Node
    # Points to src/nodes/semantic_node.py
    semantic_node = ExecuteProcess(
        cmd=['python3', 'src/nodes/semantic_node.py'],
        cwd=project_root, 
        output='screen'
    )

    return LaunchDescription([
        orbbec_launch,
        static_tf,
        TimerAction(period=3.0, actions=[rgbd_odometry]), 
        TimerAction(period=6.0, actions=[semantic_node])  
    ])