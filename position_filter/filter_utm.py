import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import NavSatFix
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
from functools import partial
from collections import deque
from tf2_ros import TransformBroadcaster
from scipy.linalg import sqrtm
from utm import from_latlon, to_latlon
from math import radians, atan2

class PositionFilter(Node):

    def __init__(self):
        super().__init__('position_filter')

        # Parameters
        leader1_gps_topic = self.declare_parameter("leader1_gps_topic", "leader1/gps").get_parameter_value().string_value
        leader2_gps_topic = self.declare_parameter("leader2_gps_topic", "leader2/gps").get_parameter_value().string_value
        leader1_distance_topic = self.declare_parameter("leader1_distance_topic", "leader1/distance").get_parameter_value().string_value
        leader2_distance_topic = self.declare_parameter("leader2_distance_topic", "leader2/distance").get_parameter_value().string_value
        follower_depth_topic = self.declare_parameter("follower_depth_topic", "lolo/smarc/depth").get_parameter_value().string_value
        follower_heading_topic = self.declare_parameter("follower_heading_topic", "lolo/smarc/heading").get_parameter_value().string_value
        follower_gps_topic = self.declare_parameter("follower_gps_topic", "lolo/standard/navsatfix").get_parameter_value().string_value
        state_topic = self.declare_parameter("state_topic", "ukf/state").get_parameter_value().string_value
        covariance_topic = self.declare_parameter("covariance_topic", "ukf/covariance").get_parameter_value().string_value
        state_gps_topic = self.declare_parameter("state_gps_topic", "ukf/gps").get_parameter_value().string_value
        self.parent_frame = self.declare_parameter("parent_frame", "leader1/base_link").get_parameter_value().string_value
        self.child_frame = self.declare_parameter("child_frame", "follower/ukf_link").get_parameter_value().string_value
        self.Q_std = self.declare_parameter("Q_std", 0.03).get_parameter_value().double_value
        self.R_std = self.declare_parameter("R_std", 0.1).get_parameter_value().double_value
        self.heading_std = self.declare_parameter("heading_std", 0.03).get_parameter_value().double_value
        self.debug_log = self.declare_parameter("debug_log", True).get_parameter_value().bool_value

        # Offsets
        self.leader1_offset = np.array([0., 0.])
        self.leader2_offset = np.array([0., 0.])
        self.follower_depth = 0.0
        self.follower_heading = 0.0

        # UTM reference zone (will be set from first GPS message)
        self.utm_zone_number = None
        self.utm_zone_letter = None

        # Subscribers containing the range of the follower from the leaders
        self.leader1_gps_sub = self.create_subscription(NavSatFix, leader1_gps_topic, partial(self.leader_gps_cb, offset=self.leader1_offset), 1)
        self.leader2_gps_sub = self.create_subscription(NavSatFix, leader2_gps_topic, partial(self.leader_gps_cb, offset=self.leader2_offset), 1)
        self.leader1_distance_sub = self.create_subscription(Float32, leader1_distance_topic, partial(self.measurement_cb, offset=self.leader1_offset), 1)
        self.leader2_distance_sub = self.create_subscription(Float32, leader2_distance_topic, partial(self.measurement_cb, offset=self.leader2_offset), 1)
        self.follower_depth_sub = self.create_subscription(Float32, follower_depth_topic, self.follower_depth_cb, 1)
        self.follower_heading_sub = self.create_subscription(Float32, follower_heading_topic, partial(self.measurement_cb, offset=np.array([0])), 1)
        self.follower_gps_sub = self.create_subscription(NavSatFix, follower_gps_topic, self.follower_gps_cb, 1)  # Only needed to set the initial state        

        # Publisher to publish the filtered data of the follower
        self.state_pub = self.create_publisher(Float32MultiArray, state_topic, 1)
        self.covariance_pub = self.create_publisher(Float32MultiArray, covariance_topic, 1)
        self.state_gps_pub = self.create_publisher(NavSatFix, state_gps_topic, 1)

        self.initialize_filter()
        
        # Timer for predict step
        # self.predict_timer = self.create_timer(self.dt, self.fake_predict_step)

        # Transform broadcaster for dynamic transform publishing
        self.tf_broadcaster = TransformBroadcaster(self)

        ns = self.get_namespace()
        if ns == "/":
            ns = "follower"     # Default namespace if none is provided
        if self.child_frame.startswith("NS"):
            self.child_frame = self.child_frame.replace("NS", ns)


        self.rolling_pos_x = deque(maxlen=4)
        self.rolling_pos_y = deque(maxlen=4)
        self.wma_weights = np.array([0.1, 0.2, 0.2, 0.5]) 

    def initialize_filter(self):
        # State: [pos_x, vel_x, pos_y, vel_y]
        # Measurement: [distance_to_leader1], or [distance_to_leader2]  
        # We dont initialize x here, default to np.zeros(dim_x)
        # So that we can compute the intial state separately
        points = MerweScaledSigmaPoints(n=4, alpha=1e-2, beta=2., kappa=-1.0)
        self.dt = 1 / 200
        self.ukf = UnscentedKalmanFilter(dim_x=4, dim_z=1, dt=self.dt, hx=None, fx=self.state_transition_function, points=points, sqrt_fn=sqrtm)    
        self.ukf.P = np.diag([0.01**2, 0.1**2, 0.05**2, 0.1**2])
        # self.ukf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=self.Q_std**2, block_size=2)
        self.ukf.R = np.array([self.R_std**2]) # Measurement noise
        self.initial_d1, self.initial_d2 = None, None
        self.prev_update_time = self.get_clock().now()

    def state_transition_function(self, x: np.ndarray, dt: float):
        pos_x, vel_x, pos_y, vel_y = x
        pos_x += vel_x * dt
        pos_y += vel_y * dt
        return np.array([pos_x, vel_x, pos_y, vel_y])
    
    # def fake_predict_step(self):
    #     # Project the state forward in time without projecting the covariance
    #     self.ukf.x = self.state_transition_function(self.ukf.x, self.dt)
    #     self.publish_state_and_covariance(self.ukf.x.tolist(), self.ukf.P.flatten().tolist())
    #     self.publish_state_gps(self.ukf.x[0], self.ukf.x[2], self.utm_zone_number, self.utm_zone_letter)

    def measurement_cb(self, msg: Float32, offset: np.ndarray):        
        # If initial state is not set, we take it from the first GPS message
        if np.array_equal(self.ukf.x, np.zeros(4)):
            return

        # Check if the filter has diverged. If yes, reinitialize.
        if self.ukf.P[0, 0] > 150.0 or self.ukf.P[2, 2] > 150.0:
            self.get_logger().warn(f'Covariance too high: {np.diag(self.ukf.P).tolist()}')
            self.get_logger().warn("Filter diverged, reinitializing...")
            self.initialize_filter()
            return

        # Assuming the above checks are passed, compute the time elapsed since the last update
        time_now = self.get_clock().now()
        time_elapsed = (time_now - self.prev_update_time).nanoseconds / 1e9
        time_elapsed = min(time_elapsed, 3.0)  # Limit time elapsed to avoid filter divergence due to too large predict covariance

        # Predict step of the filter
        self.ukf.Q = Q_discrete_white_noise(dim=2, dt=time_elapsed, var=self.Q_std**2, block_size=2)
        self.ukf.predict(dt=time_elapsed)
        self.publish_state_and_covariance(self.ukf.x.tolist(), self.ukf.P.flatten().tolist())

        # Update step of the filter
        measurement = msg.data
        if offset.shape == (2,):    # If offset is provided, we assume it's a distance measurement
            measurement = np.sqrt(measurement**2 - self.follower_depth**2)  # Adjust measurement for depth
            self.ukf.update(z=measurement, hx=partial(self.distance_measurement_function, offset=offset))
        else:   # If no offset is provided, we assume it's a heading measurement
            measurement = radians(measurement)
            self.ukf.update(z=measurement, hx=self.heading_measurement_function, R=self.heading_std**2)

        self.prev_update_time = time_now

        # Send filtered data
        self.broadcast_transform(smoothen=False)
        self.publish_state_and_covariance(self.ukf.x.tolist(), self.ukf.P.flatten().tolist())
        self.publish_state_gps(self.ukf.x[0], self.ukf.x[2], self.utm_zone_number, self.utm_zone_letter)

        if self.debug_log:
            self.get_logger().info(f'Updating with measurement: {measurement} and offset: {offset.tolist()} and time elapsed: {time_elapsed}')
            self.get_logger().info(f'Updated state: {self.ukf.x.tolist()}')
            # self.get_logger().info(f'Updated covariance: {np.diag(self.ukf.P).tolist()}')

    def leader_gps_cb(self, msg: NavSatFix, offset: np.ndarray):
        try:
            # Convert GPS to UTM coordinates
            x, y, zone_number, zone_letter = from_latlon(msg.latitude, msg.longitude)
            if self.utm_zone_number is None:
                self.utm_zone_number = zone_number
                self.utm_zone_letter = zone_letter
                self.get_logger().info(f'UTM reference zone set to: {zone_number}{zone_letter}') 
            # Update the offset for the leader
            offset[0], offset[1] = x, y
        except Exception as e:
            self.get_logger().error(f'Error converting GPS to UTM: {e}')
            return
        
    def follower_depth_cb(self, msg: Float32):
        self.follower_depth = msg.data - 1.0

    def follower_gps_cb(self, msg: NavSatFix):
        # If initial state is not set, we take it from the first GPS message
        x, y, _, _ = from_latlon(msg.latitude, msg.longitude)
        if np.array_equal(self.ukf.x, np.zeros(4)):
            self.ukf.x[0], self.ukf.x[2] = x, y  # pos_x, pos_y
            self.get_logger().info(f'Initial state set to: {self.ukf.x.tolist()}')

    def distance_measurement_function(self, x: np.ndarray, offset: np.ndarray):
        pos_x, _, pos_y, _ = x
        return np.sqrt([ (pos_x-offset[0]) ** 2 + (pos_y-offset[1]) ** 2 ])
    
    def heading_measurement_function(self, x: np.ndarray):
        _, vel_x, _, vel_y = x
        return np.array([atan2(vel_x, vel_y)])

    def publish_state_and_covariance(self, state, covariance):
        state_msg = Float32MultiArray()
        state_msg.data = state
        self.state_pub.publish(state_msg)

        covariance_msg = Float32MultiArray()
        covariance_msg.data = covariance
        self.covariance_pub.publish(covariance_msg)

    def publish_state_gps(self, x, y, zone_number=None, zone_letter=None):
        """Convert UTM coordinates back to GPS with higher precision."""
        if self.utm_zone_number is None or self.utm_zone_letter is None:
            return
        try:
            # Use higher precision for UTM to lat/lon conversion
            lat, lon = to_latlon(float(x), float(y), int(self.utm_zone_number), str(self.utm_zone_letter))
            gps_msg = NavSatFix()
            gps_msg.latitude = lat
            gps_msg.longitude = lon
            gps_msg.altitude = 0.0  # Assuming altitude is not used
            gps_msg.header.stamp = self.get_clock().now().to_msg()
            gps_msg.header.frame_id = self.child_frame
            self.state_gps_pub.publish(gps_msg)
            
        except Exception as e:
            self.get_logger().error(f'UTM to GPS conversion failed: {e}')

    def broadcast_transform(self, smoothen=False):
        pos_x, _, pos_y, _ = self.ukf.x
        if smoothen:
            pos_x = self.wma(pos_x, self.rolling_pos_x, self.wma_weights)
            pos_y = self.wma(pos_y, self.rolling_pos_y, self.wma_weights)

        # Publish dynamic transform
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self.parent_frame
        transform.child_frame_id = self.child_frame
        transform.transform.translation.x = pos_x
        transform.transform.translation.y = pos_y
        transform.transform.translation.z = 0.0  # Assuming z is 0
        transform.transform.rotation.x = 0.0
        transform.transform.rotation.y = 0.0
        transform.transform.rotation.z = 0.0
        transform.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(transform)

    def wma(self, data: float, prev_data: deque, weights: np.ndarray) -> float:
        prev_data.append(data)
        if len(prev_data) < 4:
            return data
        data_array = np.array(prev_data)
        smoothened_data = np.dot(weights, data_array)
        prev_data.append(data)
        return smoothened_data

def main(args=None):
    rclpy.init(args=args)
    position_filter = PositionFilter()
    rclpy.spin(position_filter)
    position_filter.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()