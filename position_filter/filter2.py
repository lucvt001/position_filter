import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray
from geometry_msgs.msg import TransformStamped
import numpy as np
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
from functools import partial
from collections import deque
from tf2_ros import TransformBroadcaster
from scipy.linalg import sqrtm

class PositionFilter(Node):

    def __init__(self):
        super().__init__('position_filter')

        # Parameters
        leader2_offset = self.declare_parameter("leader2_offset", -10.0).get_parameter_value().double_value
        leader1_distance_topic = self.declare_parameter("leader1_distance_topic", "distance_to_leader1").get_parameter_value().string_value
        leader2_distance_topic = self.declare_parameter("leader2_distance_topic", "distance_to_leader2").get_parameter_value().string_value
        state_topic = self.declare_parameter("state_topic", "ukf/state").get_parameter_value().string_value
        covariance_topic = self.declare_parameter("covariance_topic", "ukf/covariance").get_parameter_value().string_value
        self.parent_frame = self.declare_parameter("parent_frame", "leader1/base_link").get_parameter_value().string_value
        self.child_frame = self.declare_parameter("child_frame", "follower/ukf_link").get_parameter_value().string_value
        self.Q_std = self.declare_parameter("Q_std", 0.9).get_parameter_value().double_value
        self.R_std = self.declare_parameter("R_std", 0.3).get_parameter_value().double_value
        self.debug_log = self.declare_parameter("debug_log", False).get_parameter_value().bool_value

        # Offsets
        self.leader1_offset = np.array([0., 0.])
        self.leader2_offset = np.array([0., leader2_offset])

        # Subscribers containing the range of the follower from the leaders
        self.leader1_distance_sub = self.create_subscription(Float32, leader1_distance_topic, partial(self.distance_cb, offset=self.leader1_offset), 1)
        self.leader2_distance_sub = self.create_subscription(Float32, leader2_distance_topic, partial(self.distance_cb, offset=self.leader2_offset), 1)

        # Publisher to publish the filtered data of the follower
        self.state_pub = self.create_publisher(Float32MultiArray, state_topic, 1)
        self.covariance_pub = self.create_publisher(Float32MultiArray, covariance_topic, 1)

        # Transform broadcaster for dynamic transform publishing
        self.tf_broadcaster = TransformBroadcaster(self)

        ns = self.get_namespace()
        if ns == "/":
            ns = "follower"     # Default namespace if none is provided
        if self.child_frame.startswith("NS"):
            self.child_frame = self.child_frame.replace("NS", ns)

        self.initialize_filter()

        self.rolling_pos_x = deque(maxlen=4)
        self.rolling_pos_y = deque(maxlen=4)
        self.wma_weights = np.array([0.1, 0.2, 0.2, 0.5]) 

    def initialize_filter(self):
        # State: [pos_x, vel_x, pos_y, vel_y]
        # Measurement: [distance_to_leader1], or [distance_to_leader2]  
        # We dont initialize x here, default to np.zeros(dim_x)
        # So that we can compute the intial state separately
        points = MerweScaledSigmaPoints(n=4, alpha=1e-2, beta=2., kappa=-1.0)
        self.ukf = UnscentedKalmanFilter(dim_x=4, dim_z=1, dt=0.5, hx=None, fx=self.state_transition_function, points=points, sqrt_fn=sqrtm)    
        self.ukf.P = np.diag([6.0**2, 3.0**2, 6.0**2, 3.0**2])
        self.ukf.R = np.array([self.R_std**2]) # Measurement noise
        self.initial_d1, self.initial_d2 = None, None
        self.prev_update_time = self.get_clock().now()

    def state_transition_function(self, x: np.ndarray, dt: float):
        pos_x, vel_x, pos_y, vel_y = x
        pos_x += vel_x * dt
        pos_y += vel_y * dt
        return np.array([pos_x, vel_x, pos_y, vel_y])

    def distance_cb(self, msg: Float32, offset: np.ndarray):
        # Compute the initial estimate of the state using the triangulation method
        if np.array_equal(self.ukf.x, np.zeros(4)):
            if self.initial_d1 is None and np.array_equal(offset, self.leader1_offset):
                self.initial_d1 = msg.data
            elif self.initial_d2 is None and np.array_equal(offset, self.leader2_offset):
                self.initial_d2 = msg.data
            elif self.initial_d1 is not None and self.initial_d2 is not None:
                pos_x, pos_y = self.triangulate(np.abs(self.leader2_offset[1]), self.initial_d1, self.initial_d2)
                self.ukf.x = np.array([pos_x, 0., pos_y, 0.])
                self.get_logger().info(f'Initial state set: {self.ukf.x.tolist()}')
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
        time_elapsed = min(time_elapsed, 3.0)  # Limit time elapsed to 3 seconds to avoid filter divergence due to too large predict covariance

        # Predict step of the filter
        self.ukf.Q = Q_discrete_white_noise(dim=2, dt=time_elapsed, var=self.Q_std**2, block_size=2)
        self.ukf.predict(dt=time_elapsed)
        # self.publish_state_and_covariance()
        if self.debug_log:
            self.get_logger().info(f'Predicted state: {self.ukf.x.tolist()} with dt: {time_elapsed}')
            self.get_logger().info(f'Predicted covariance: {np.diag(self.ukf.P).tolist()}')

        # Update step of the filter
        measurement = msg.data
        self.ukf.update(z=measurement, hx=partial(self.measurement_function, offset=offset))
        self.prev_update_time = time_now

        # Send filtered data
        self.broadcast_transform(smoothen=False)
        self.publish_state_and_covariance()

        if self.debug_log:
            self.get_logger().info(f'Updating with measurement: {measurement} and offset: {offset.tolist()} and time elapsed: {time_elapsed}')
            self.get_logger().info(f'Updated state: {self.ukf.x.tolist()}')
            self.get_logger().info(f'Updated covariance: {np.diag(self.ukf.P).tolist()}')

    def measurement_function(self, x: np.ndarray, offset: np.ndarray):
        pos_x, _, pos_y, _ = x
        return np.sqrt([ (pos_x-offset[0]) ** 2 + (pos_y-offset[1]) ** 2 ])
    
    def triangulate(self, d, d1, d2):
        # Given a triangle with sides d1, d2 and distance d between the two points
        # Calculate the coordinates of the third point in FLU frame
        # Using the Law of Cosines to find the angle between d and d1
        cosine = (d**2 + d1**2 - d2**2) / (2 * d * d1)
        angle = np.arccos(cosine)
        sine = np.sin(angle)
        pos_x, pos_y = -d1*sine , -d1*cosine     # FLU frame
        return (pos_x, pos_y)

    def publish_state_and_covariance(self):
        state = Float32MultiArray()
        state.data = self.ukf.x.tolist()
        self.state_pub.publish(state)

        covariance = Float32MultiArray()
        covariance.data = self.ukf.P.flatten().tolist()
        self.covariance_pub.publish(covariance)

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