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

class PositionFilter(Node):

    def __init__(self):
        super().__init__('position_filter')

        # Initialize class variables
        self.prev_heading, self.current_heading = None, None
        self.rpm = 0
        self.horizontal_thrust_vector = 0.
        self.dt = 0.1

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
        self.initial_estimate = self.declare_parameter("initial_estimate", [-10.0, 0.0, -2.0, 0.0]).get_parameter_value().double_array_value

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

        # Timer for predict step of the filter
        self.timer = self.create_timer(self.dt, self.filter_predict)

        # Initialize the filter
        # State: [pos_x, vel_x, pos_y, vel_y]
        # Measurement: [distance_to_leader1], or [distance_to_leader2]  

        self.filter_reset()
        self.prev_update_time = self.get_clock().now()

        self.rolling_pos_x = deque(maxlen=4)
        self.rolling_pos_y = deque(maxlen=4)
        self.wma_weights = np.array([0.1, 0.2, 0.2, 0.5])

    def filter_reset(self):
        points = MerweScaledSigmaPoints(n=4, alpha=1e-2, beta=2., kappa=-1.0)
        self.ukf = UnscentedKalmanFilter(dim_x=4, dim_z=1, dt=self.dt, points=points, fx=self.state_transition_function, hx=None)    
        self.ukf.x = np.array(self.initial_estimate)  # Initial state estimate
        self.ukf.P = np.diag([9., 9., 9., 9.])
        self.ukf.Q = Q_discrete_white_noise(dim=2, dt=self.dt, var=self.Q_std**2, block_size=2)
        self.ukf.R = np.array([self.R_std**2]) # Measurement noise
        self.get_logger().info(f'UKF initialized with state: {self.ukf.x.tolist()}')

    def filter_predict(self):
        P = self.ukf.P.diagonal()
        if P[0] > 20 and P[2] > 20:  # Stop predicting if the covariance is too large because it means there has been no update for a while
            return
        self.ukf.predict()
        self.publish_state_and_covariance()       
        # self.get_logger().info(f'UKF state: {self.ukf.x.tolist()}')

    def state_transition_function(self, x: np.ndarray, dt: float):
        pos_x, vel_x, pos_y, vel_y = x
        pos_x += vel_x * dt
        pos_y += vel_y * dt
        return np.array([pos_x, vel_x, pos_y, vel_y])

    def distance_cb(self, msg: Float32, offset: np.ndarray):
        time_now = self.get_clock().now()
        time_elapsed = (time_now - self.prev_update_time).nanoseconds / 1e9
        if time_elapsed <= 0.11:    # to avoid updating the filter many times at once which can cause it to collapse
            return 
        self.prev_update_time = time_now
        
        measurement = msg.data
        self.ukf.hx = partial(self.measurement_function, offset=offset)
        self.ukf.update(z=measurement)

        self.smoothen_state_and_broadcast_transform()
        self.publish_state_and_covariance()

    def measurement_function(self, x: np.ndarray, offset: np.ndarray):
        pos_x, _, pos_y, _ = x
        return np.sqrt([ (pos_x-offset[0]) ** 2 + (pos_y-offset[1]) ** 2 ])

    def publish_state_and_covariance(self):
        state = Float32MultiArray()
        state.data = self.ukf.x.tolist()
        self.state_pub.publish(state)

        covariance = Float32MultiArray()
        covariance.data = self.ukf.P.flatten().tolist()
        self.covariance_pub.publish(covariance)

    def smoothen_state_and_broadcast_transform(self):
        pos_x, _, pos_y, _ = self.ukf.x
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