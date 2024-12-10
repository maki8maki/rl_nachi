from typing import Optional, Type

import cv2
import numpy as np
import rospy
import rotations as rot
import tf
from cv_bridge import CvBridge, CvBridgeError
from gymnasium import spaces
from nachi_opennr_msgs.srv import (
    TriggerWithResultCode,
    getGeneralSignal,
    getGeneralSignalResponse,
    setGeneralSignal,
)
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray

# OpenNR_IFに関連する定数
ANGLE_COMMAND_TOPIC_NAME = "/joint_group_angle_controller/command"
POSITION_COMMAND_TOPIC_NAME = "/joint_group_position_controller/command"
RGB_IMAGE_TOPIC_NAME = "/camera/camera/color/image_rect_raw"
DEPTH_IMAGE_TOPIC_NAME = "/camera/camera/aligned_depth_to_color/image_raw"
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
TOOL_LINK_NAME = "tool_link"

# Depthカメラに関連する定数
DEPTH_MIN = 70  # mm
DEPTH_NAX = 500  # mm

# 表示に関する定数
IMAGE_MIN = 0
IMAGE_MAX = 255
IMAGE_WIDTH = 848
IMAGE_HEIGHT = 480

# マニピュレータの制限に関する定数
SHIFT_MIN = np.array([200.0, -150.0, 300.0, -30.0, -30.0, -30.0])  # mm, deg
# SHIFT_MIN = np.array([200.0, -150.0, 350.0, -30.0, -30.0, -30.0])  # mm, deg
SHIFT_MAX = np.array([600.0, 150.0, 450.0, 30.0, 50.0, 30.0])  # mm, deg


class NachiEnv:
    def __init__(self):
        # 位置姿勢取得の準備
        self.tf_listener = tf.TransformListener()
        self.tool_pose: np.ndarray = np.zeros((6,), dtype=np.float64)

        # 画像取得・表示の準備
        self.bridge = CvBridge()
        # cv2.namedWindow("Images")
        self.rgb_image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        self.depth_image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
        self.rgb_image_sub = rospy.Subscriber(RGB_IMAGE_TOPIC_NAME, Image, self.rgb_image_callback, queue_size=1)
        self.depth_image_sub = rospy.Subscriber(DEPTH_IMAGE_TOPIC_NAME, Image, self.depth_image_callback, queue_size=1)

        # 動作指令の準備
        self.angle_command_pub = rospy.Publisher(ANGLE_COMMAND_TOPIC_NAME, Float64MultiArray, queue_size=1)
        self.position_command_pub = rospy.Publisher(POSITION_COMMAND_TOPIC_NAME, Float64MultiArray, queue_size=1)

        # その他
        self.rate = rospy.Rate(100)  # 諸々の待機に利用
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,))
        self.observation_space = spaces.Box(
            low=np.array([-0.0, -1.0, -0.0, -np.pi, -np.pi, -np.pi]),
            high=np.array([2.0, 1.0, 2.0, np.pi, np.pi, np.pi]),
            dtype="float64",
        )

        self.check_all_systems_ready()

        # 接続開始
        response = self.call_service(SRV_NAME_OPEN, TriggerWithResultCode)
        assert response is not None

        # モータをオンにする
        response = self.call_service(SRV_NAME_CTRLMOTER_ON, TriggerWithResultCode)
        assert response is not None

    def rgb_image_callback(self, data: Image):
        try:
            bgr_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            # self.update_display(bgr_image=bgr_image)
        except CvBridgeError as e:
            rospy.logerr(f"Failed to convert RGB image: {e}")

    def depth_image_callback(self, data: Image):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            depth_image[(depth_image > DEPTH_NAX) | (depth_image < DEPTH_MIN)] = 0
            depth_image = depth_image * 255.0 / DEPTH_NAX
            self.depth_image = depth_image.astype(np.uint8)
            self.update_display()
        except CvBridgeError as e:
            rospy.logerr(f"Failed to convert Depth image: {e}")

    def update_display(self, bgr_image: Optional[np.ndarray] = None):
        if bgr_image is None:
            bgr_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_RGB2BGR)
        depth_image = cv2.cvtColor(self.depth_image, cv2.COLOR_GRAY2BGR)
        merged_image = np.vstack((bgr_image, depth_image))
        cv2.imshow("Images", merged_image)
        cv2.waitKey(1)

    def check_all_systems_ready(self):
        rospy.logdebug("Waiting for all systems to be ready...")
        self.check_all_sensors_ready()
        self.check_transform_ready()
        self.check_publishers_connection()
        rospy.loginfo("All Systems Ready")

    def check_all_sensors_ready(self):
        rospy.logdebug("Waiting for all sensros to be ready...")
        self.check_rgb_image_ready()
        self.check_depth_image_ready()
        rospy.loginfo("All Sensors Ready")

    def check_rgb_image_ready(self):
        rospy.logdebug(f"Waiting for {RGB_IMAGE_TOPIC_NAME} to be ready...")
        rospy.wait_for_message(RGB_IMAGE_TOPIC_NAME, Image)
        # self.rgb_image_callback(data)
        rospy.logdebug(f"{RGB_IMAGE_TOPIC_NAME} Ready")

    def check_depth_image_ready(self):
        rospy.logdebug(f"Waiting for {DEPTH_IMAGE_TOPIC_NAME} to be ready...")
        rospy.wait_for_message(DEPTH_IMAGE_TOPIC_NAME, Image)
        # self.depth_image_callback(data)
        rospy.logdebug(f"{DEPTH_IMAGE_TOPIC_NAME} Ready")

    def check_transform_ready(self):
        rospy.logdebug(f"Waiting for transform from {BASE_LINK_NAME} to {TOOL_LINK_NAME} to be ready...")
        self.tf_listener.waitForTransform(BASE_LINK_NAME, TOOL_LINK_NAME, rospy.Time(), rospy.Duration(2.0))
        rospy.logdebug(f"transform from {BASE_LINK_NAME} to {TOOL_LINK_NAME} Ready")
        self.update_robot_state()
        rospy.loginfo("All Transform Ready")

    def check_publishers_connection(self):
        rospy.logdebug("Waiting for all publishers to be connected...")
        self.check_angle_command_pub_connection()
        self.check_position_command_pub_connection()
        rospy.loginfo("All Publishers Ready")

    def check_angle_command_pub_connection(self):
        rospy.logdebug("Waiting for angle_command_pub to be connected...")
        while self.angle_command_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            self.rate.sleep()
        rospy.logdebug("angle_command_pub Connected")

    def check_position_command_pub_connection(self):
        rospy.logdebug("Waiting for position_command_pub to be connected...")
        while self.position_command_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            self.rate.sleep()
        rospy.logdebug("position_command_pub Connected")

    def update_robot_state(self):
        now = rospy.Time.now()
        self.tf_listener.waitForTransform(BASE_LINK_NAME, TOOL_LINK_NAME, now, rospy.Duration(2.0))
        try:
            # tool
            (trans, quat) = self.tf_listener.lookupTransform(BASE_LINK_NAME, TOOL_LINK_NAME, now)
            self.tool_pose[:3] = np.array(trans, dtype=np.float64)  # m

            # rosとmujocoでクォータニオンの形式が異なるので変換
            m_quat = [quat[3]]
            m_quat += quat[:3]

            euler = rot.quat2euler(m_quat)  # rad
            self.tool_pose[3:] = np.array(euler, dtype=np.float64)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logdebug(f"Error during update_robot_state: {e}")

    def set_action(self, action: np.ndarray):
        self.update_robot_state()

        action = np.clip(action.copy(), -1, 1)

        # unscale
        pos_ctrl, rot_ctrl = action[:3], action[3:]
        pos_ctrl *= 0.05  # m
        rot_ctrl *= np.deg2rad(10)  # rad

        # 目標の計算
        pos_cur, quat_cur = self.tool_pose[:3], self.tool_pose[3:]
        pos_target = pos_cur + pos_ctrl
        mat_target = rot.add_rot_mat(rot.euler2mat(quat_cur), rot.euler2mat(rot_ctrl))
        rot_target = rot.mat2euler(mat_target)

        # z軸周りがraw、x軸周りがyawと定義されている
        rot_target[0], rot_target[2] = rot_target[2], rot_target[0]
        target = np.concatenate([pos_target * 1000, np.rad2deg(rot_target)])

        # 指令の送信
        self.set_position_action(target)

    def grasp(self, z_diff=-0.002):
        self.update_robot_state()
        diff = np.array([0.0, 0.0, z_diff])
        mat = rot.euler2mat(self.tool_pose[3:])
        tgt_diff = mat @ diff
        pos_target = self.tool_pose[:3] + tgt_diff
        rot_target = self.tool_pose[3:]
        rot_target[0], rot_target[2] = rot_target[2], rot_target[0]
        target = np.concatenate([pos_target * 1000, np.rad2deg(rot_target)])
        self.set_position_action(target)

    def set_angle_action(self, target: np.ndarray):
        assert target.shape == (6,)
        msg = Float64MultiArray()
        msg.data = target
        self.angle_command_pub.publish(msg)
        rospy.logdebug(f"Published angle target: {target}")

        self.wait_action()

    def set_position_action(self, target: np.ndarray):
        assert target.shape == (6,)
        assert (SHIFT_MIN <= target).all() and (
            target <= SHIFT_MAX
        ).all(), f"position target is out of range. target: {target}"
        msg = Float64MultiArray()
        msg.data = target
        self.position_command_pub.publish(msg)
        rospy.logdebug(f"Published position target: {target}")

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
                1.53,
                -0.401,
                0.0,
                -1.17,
                0.0,
            ]
        )
        self.set_angle_action(target)

    def set_valve_state(self, state: int):
        """
        state: int, If  state is1, valve on (= suction), else if state is 0 valve off
        """
        assert state == 1 or state == 0, f"state must be 1 or 0, state: {state}"

        self.call_service(
            SRV_NAME_SET_ACSGENERAL_OUTPUT_SIGNAL, setGeneralSignal, [state], VALVE_GENERAL_OUTPUT_NUMBER, 1
        )

    def set_valve_on(self):
        self.set_valve_state(1)

    def set_valve_off(self):
        self.set_valve_state(0)

    def is_moving(self) -> bool:
        response = self.call_service(
            SRV_NAME_GET_ACSGENERAL_OUTPUT_SIGNAL, getGeneralSignal, MOVING_GENERAL_OUTPUT_NUMBER, 1
        )
        if isinstance(response, getGeneralSignalResponse):
            return response.signal_state[0] == 1
        else:
            # サービスを取得できない場合は動作中にする
            return True

    def wait_action(self):
        # 指令の受付まで待機、一定時間で先に進む
        start = rospy.Time.now()
        while not self.is_moving() and (rospy.Time.now() - start) <= rospy.Duration(0.5):
            self.rate.sleep()

        # 動作の終了まで待機
        while self.is_moving():
            self.rate.sleep()

    def call_service(self, service_name: str, service_class: Type[object], *call_args):
        rospy.wait_for_service(service_name)
        try:
            client = rospy.ServiceProxy(service_name, service_class)
            response = client(*call_args)
            assert response.result >= NR_E_NORMAL, f"{service_name} return error code: {response.result}"
            return response
        except rospy.ServiceException:
            rospy.logwarn(f"Service call failed: {service_name}")
            return None

    def close(self):
        # バルブをオフにする
        self.set_valve_off()

        # 待機位置に移動する
        self.set_waiting_position()

        # モータをオフにする
        self.call_service(SRV_NAME_CTRLMOTER_OFF, TriggerWithResultCode)

        # 接続終了
        self.call_service(SRV_NAME_CLOSE, TriggerWithResultCode)
