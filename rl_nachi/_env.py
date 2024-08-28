from typing import Type

import cv2
import numpy as np
import rclpy
import rclpy.time
import tf2_ros
from cv_bridge import CvBridge, CvBridgeError
from gymnasium import spaces
from nachi_opennr_msgs.srv import (
    GetGeneralSignal,
    SetGeneralSignal,
    TriggerWithResultCode,
)
from rclpy.node import Node
from realsense2_camera_msgs.msg import RGBD
from std_msgs.msg import Float64MultiArray
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from . import rotations as rot

# OpenNR_IFに関連する定数
ANGLE_COMMAND_TOPIC_NAME = "/joint_group_angle_controller/commands"
POSITION_COMMAND_TOPIC_NAME = "/joint_group_position_controller/commands"
SRV_NAME_OPEN = "nachi_open"
SRV_NAME_CLOSE = "nachi_close"
SRV_NAME_CTRLMOTER_ON = "nachi_ctrl_motor_on"
SRV_NAME_CTRLMOTER_OFF = "nachi_ctrl_motor_off"
SRV_NAME_GET_ACSGENERAL_OUTPUT_SIGNAL = "nachi_get_acs_general_output_signal"
SRV_NAME_SET_ACSGENERAL_OUTPUT_SIGNAL = "nachi_set_acs_general_output_signal"
VALVE_GENERAL_OUTPUT_NUMBER = 1
MOVING_GENERAL_OUTPUT_NUMBER = 26
NR_E_NORMAL = 0

# TFに関連する定数
BASE_LINK_NAME = "base_link"
FLANGE_LINK_NAME = "flange_link"
TOOL_LINK_NAME = "tool_link"

# RGB-Dカメラに関連する定数
RGBD_IMAGE_TOPIC_NAME = "/camera/camera/rgbd"
DEPTH_MIN = 70  # mm
DEPTH_NAX = 500  # mm
IMAGE_MIN = 0
IMAGE_MAX = 255
IMAGE_WIDTH = 848
IMAGE_HEIGHT = 480

# マニピュレータの制限に関する定数
SHIFT_MIN = np.array([130.0, -300.0, 350.0, -30.0, -45.0, -90.0])  # mm, deg
SHIFT_MAX = np.array([580.0, 300.0, 600.0, 30.0, 90.0, 90.0])  # mm, deg


class NachiEnv(Node):
    def __init__(self):
        super().__init__(self.__class__.__name__)

        # 接続開始
        response = self.call_service(
            SRV_NAME_OPEN, TriggerWithResultCode, TriggerWithResultCode.Request(), use_future=True
        )
        assert response is not None

        # モータをオンにする
        response = self.call_service(
            SRV_NAME_CTRLMOTER_ON, TriggerWithResultCode, TriggerWithResultCode.Request(), use_future=True
        )
        assert response is not None

        # 位置姿勢取得の準備
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tool_pose = np.zeros((6,), dtype=np.float64)
        self.flange_pose = np.zeros((7,), dtype=np.float64)  # x, y, z, quat
        self.tf_timer = self.create_timer(1.0 / 100, self.update_robot_state)

        # 画像取得・表示の準備
        self.bridge = CvBridge()
        cv2.namedWindow("Images")
        self.rgb_image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        self.depth_image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
        self.rgbd_image_sub = self.create_subscription(RGBD, RGBD_IMAGE_TOPIC_NAME, self.rgbd_image_callback, 1)

        # 動作指令の準備
        self.angle_command_pub = self.create_publisher(Float64MultiArray, ANGLE_COMMAND_TOPIC_NAME, 1)
        self.position_command_pub = self.create_publisher(Float64MultiArray, POSITION_COMMAND_TOPIC_NAME, 1)

        # その他
        self.rate = self.create_rate(100)  # 諸々の待機に利用
        self.robot_act_dim = 6
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.robot_act_dim,))
        self.observation_space = spaces.Box(
            low=np.array([-0.0, -1.0, -0.0, -np.pi, -np.pi, -np.pi]),
            high=np.array([2.0, 1.0, 2.0, np.pi, np.pi, np.pi]),
            dtype="float64",
        )

        self.check_all_systems_ready()

        self.running = True

    def rgbd_image_callback(self, data: RGBD):
        try:
            # rgb
            bgr_image = self.bridge.imgmsg_to_cv2(data.rgb, "bgr8")
            self.rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

            # depth
            depth_image = self.bridge.imgmsg_to_cv2(data.depth, data.depth.encoding)
            depth_image[(depth_image > DEPTH_NAX) | (depth_image < DEPTH_MIN)] = 0
            depth_image = depth_image * 255.0 / DEPTH_NAX
            self.depth_image = depth_image.astype(np.uint8)

            print(depth_image.shape)

            self.update_display(bgr_image=bgr_image)
        except CvBridgeError as e:
            self.get_logger().error(f"Failed to convert RGBD image: {e}")

    def update_display(self, bgr_image: np.ndarray):
        depth_image = cv2.cvtColor(self.depth_image, cv2.COLOR_GRAY2BGR)
        merged_image = np.vstack((bgr_image, depth_image))
        cv2.imshow("Images", merged_image)
        cv2.waitKey(1)

    def check_all_systems_ready(self):
        self.get_logger().debug("Waiting for all systems to be ready...")
        self.check_all_sensors_ready()
        self.check_transform_ready()
        self.check_publishers_connection()
        self.get_logger().info("All Systems Ready")

    def check_all_sensors_ready(self):
        self.get_logger().debug("Waiting for all sensros to be ready...")
        self.check_rgbd_image_ready()
        self.get_logger().info("All Sensors Ready")

    def check_rgbd_image_ready(self):
        self.get_logger().debug(f"Waiting for {RGBD_IMAGE_TOPIC_NAME} to be ready...")
        self.wait_for_topic_published(RGBD_IMAGE_TOPIC_NAME)
        self.get_logger().debug(f"{RGBD_IMAGE_TOPIC_NAME} Ready")

    def check_transform_ready(self):
        self.get_logger().debug(f"Waiting for transform from {BASE_LINK_NAME} to {TOOL_LINK_NAME} to be ready...")
        future = self.tf_buffer.wait_for_transform_async(BASE_LINK_NAME, TOOL_LINK_NAME, rclpy.time.Time())
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        self.get_logger().debug(f"transform from {BASE_LINK_NAME} to {TOOL_LINK_NAME} Ready")
        self.update_robot_state()
        self.get_logger().info("All Transform Ready")

    def check_publishers_connection(self):
        self.get_logger().debug("Waiting for all publishers to be connected...")
        self.check_angle_command_pub_connection()
        self.check_position_command_pub_connection()
        self.get_logger().info("All Publishers Ready")

    def check_angle_command_pub_connection(self):
        self.get_logger().debug("Waiting for angle_command_pub to be connected...")
        while self.angle_command_pub.get_subscription_count == 0 and rclpy.ok():
            self.rate.sleep()
        self.get_logger().debug("angle_command_pub Connected")

    def check_position_command_pub_connection(self):
        self.get_logger().debug("Waiting for position_command_pub to be connected...")
        while self.position_command_pub.get_subscription_count == 0 and rclpy.ok():
            self.rate.sleep()
        self.get_logger().debug("position_command_pub Connected")

    def update_robot_state(self):
        now = self.get_clock().now()
        future = self.tf_buffer.wait_for_transform_async(BASE_LINK_NAME, TOOL_LINK_NAME, now)
        rclpy.spin_until_future_complete(self, future, timeout_sec=0.5)
        try:
            # tool
            data = self.tf_buffer.lookup_transform(BASE_LINK_NAME, TOOL_LINK_NAME, now)
            trans = data.transform.translation
            trans = [trans.x, trans.y, trans.z]
            quat = data.transform.rotation
            quat = [quat.w, quat.x, quat.y, quat.z]  # rosとmujocoでクォータニオンの形式が異な

            euler = rot.quat2euler(quat)  # rad

            self.tool_pose[:3] = np.array(trans, dtype=np.float64)  # m
            self.tool_pose[3:] = np.array(euler, dtype=np.float64)

            # flange
            data = self.tf_buffer.lookup_transform(BASE_LINK_NAME, FLANGE_LINK_NAME, now)
            trans = data.transform.translation
            trans = [trans.x, trans.y, trans.z]
            self.flange_pose[:3] = np.array(trans, dtype=np.float64)
            self.flange_pose[3:] = np.array(quat, dtype=np.float64) * -1.0
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().debug(f"Error during update_robot_state: {e}")

    def set_action(self, action: np.ndarray):
        action = np.clip(action.copy(), -1, 1)

        # unscale
        pos_ctrl, rot_ctrl = action[:3], action[3:]
        pos_ctrl *= 0.05  # m
        rot_ctrl *= np.deg2rad(10)  # rad

        # 目標の計算（flangeとtoolは固定されている）
        pos_cur, quat_cur = self.flange_pose[:3], self.flange_pose[3:]
        pos_target = pos_cur + pos_ctrl
        mat_target = rot.add_rot_mat(rot.quat2mat(quat_cur), rot.euler2mat(rot_ctrl))
        rot_target = rot.mat2euler(mat_target)

        # z軸周りがroll、x軸周りがyawと定義されている
        rot_target[0], rot_target[2] = rot_target[2], rot_target[0]
        target = np.concatenate([pos_target * 1000, np.rad2deg(rot_target)])

        # 指令の送信
        self.set_position_action(target)

    def set_angle_action(self, target: np.ndarray):
        assert target.shape == (6,)
        msg = Float64MultiArray(data=target)
        self.angle_command_pub.publish(msg)
        self.get_logger().debug(f"Published angle target: {target}")

        self.wait_action()

    def set_position_action(self, target: np.ndarray):
        assert target.shape == (6,)
        assert (SHIFT_MIN <= target).all() and (target <= SHIFT_MAX).all(), target
        msg = Float64MultiArray(data=target)
        self.position_command_pub.publish(msg)
        self.get_logger().debug(f"Published position target: {target}")

        self.wait_action()

    def set_waiting_position(self):
        target = np.array(
            [
                0.0,
                90.0,
                0.0,
                0.0,
                -90.0,
                0.0,
            ]
        )
        self.set_angle_action(target)

    def set_initial_position(self):
        target = np.rad2deg(
            [
                0.0,
                1.67,
                -0.157,
                0.0,
                -1.57,
                0.0,
            ]
        )
        self.set_angle_action(target)

    def set_valve_state(self, state: int):
        """
        state: int, If  state is1, valve on (= suction), else if state is 0 valve off
        """
        assert state == 1 or state == 0, f"state must be 1 or 0, state: {state}"

        req = SetGeneralSignal.Request()
        req.value = [state]
        req.subid = VALVE_GENERAL_OUTPUT_NUMBER
        req.count = 1
        self.call_service(SRV_NAME_SET_ACSGENERAL_OUTPUT_SIGNAL, SetGeneralSignal, req)

    def set_valve_on(self):
        self.set_valve_state(1)

    def set_valve_off(self):
        self.set_valve_state(0)

    def is_moving(self) -> bool:
        req = GetGeneralSignal.Request()
        req.subid = MOVING_GENERAL_OUTPUT_NUMBER
        req.count = 1
        response = self.call_service(SRV_NAME_GET_ACSGENERAL_OUTPUT_SIGNAL, GetGeneralSignal, req)
        if isinstance(response, GetGeneralSignal.Response):
            return response.signal_state[0] == 1
        else:
            # サービスを取得できない場合は動作中にする
            return True

    def wait_action(self):
        # 指令の受付まで待機、一定時間で先に進む
        start = self.get_clock().now()
        while not self.is_moving() and (self.get_clock().now() - start) <= rclpy.time.Duration(0.5):
            self.rate.sleep()

        # 動作の終了まで待機
        while self.is_moving():
            self.rate.sleep()

    def call_service(self, service_name: str, service_class: Type[object], request: object, use_future: bool = False):
        client = self.create_client(service_class, service_name)
        try:
            while not client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info("service not available, waiting again...")
            if use_future:
                future = client.call_async(request)
                rclpy.spin_until_future_complete(self, future)
                response = future.result()
            else:
                response = client.call(request)
            assert response.result >= NR_E_NORMAL, f"{service_name} return error code: {response.result}"
            return response
        except Exception as e:
            self.get_logger().warn(f"Service call failed: {service_name}, message: {str(e)}")
            return None

    def wait_for_topic_published(self, topic_name: str):
        while True:
            topic_list = self.get_topic_names_and_types()
            for name, _ in topic_list:
                if name == topic_name:
                    return
            self.rate.sleep()

    def close(self):
        # バルブをオフにする
        self.set_valve_off()

        # 待機位置に移動する
        self.set_waiting_position()

        # モータをオフにする
        self.call_service(SRV_NAME_CTRLMOTER_OFF, TriggerWithResultCode, TriggerWithResultCode.Request())

        # 接続終了
        self.call_service(SRV_NAME_CLOSE, TriggerWithResultCode, TriggerWithResultCode.Request())

        # ノードの終了
        self.destroy_node()


if __name__ == "__main__":
    rclpy.init()
    env = NachiEnv()
    try:
        rclpy.spin(env)
    except KeyboardInterrupt:
        env.close()

    rclpy.shutdown()
