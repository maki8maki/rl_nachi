from typing import Type

import cv2
import numpy as np
import rospy
import tf
from cv_bridge import CvBridge, CvBridgeError
from gymnasium import spaces
from nachi_opennr_msgs.srv import (
    TriggerWithResultCode,
    getGeneralSignal,
    getGeneralSignalResponse,
)
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray

from . import rotations as rot

# OpenNR_IFに関連する定数
POSITION_COMMAND_TOPIC_NAME = "/joint_group_position_controller/command"
RGB_IMAGE_TOPIC_NAME = "/camera/camera/color/image_rect_raw"
DEPTH_IMAGE_TOPIC_NAME = "/camera/camera/aligned_depth_to_color/image_raw"
SRV_NAME_OPEN = "nachi_open"
SRV_NAME_CLOSE = "nachi_close"
SRV_NAME_CTRLMOTER_ON = "nachi_ctrl_motor_on"
SRV_NAME_CTRLMOTER_OFF = "nachi_ctrl_motor_off"
SRV_NAME_GET_ACSGENERAL_OUTPUT_SIGNAL = "nachi_get_acs_general_output_signal"
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


class NachiEnv:
    def __init__(self):
        # 接続開始
        response = self.call_service(SRV_NAME_OPEN, TriggerWithResultCode)
        assert response is not None

        # モータをオンにする
        response = self.call_service(SRV_NAME_CTRLMOTER_ON, TriggerWithResultCode)
        assert response is not None

        # 位置姿勢取得の準備
        self.tf_listener = tf.TransformListener()
        self.robot_state = np.zeros((6,), dtype=np.float64)
        self.update_robot_state()

        # 画像取得・表示の準備
        self.bridge = CvBridge()
        cv2.namedWindow("Images")
        self.rgb_image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        self.depth_image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
        self.rgb_image_sub = rospy.Subscriber(RGB_IMAGE_TOPIC_NAME, Image, self.rgb_image_callback, queue_size=1)
        self.depth_image_sub = rospy.Subscriber(DEPTH_IMAGE_TOPIC_NAME, Image, self.depth_image_callback, queue_size=1)

        # 動作指令の準備
        self.position_command_pub = rospy.Publisher(POSITION_COMMAND_TOPIC_NAME, Float64MultiArray, queue_size=1)

        # その他
        self.rate = rospy.Rate(100)  # 諸々の待機に利用
        self.robot_act_dim = 6
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.robot_act_dim,))
        self.observation_space = spaces.Box(
            low=np.array([-0.0, -1.0, -0.0, -np.pi, -np.pi, -np.pi]),
            high=np.array([2.0, 1.0, 2.0, np.pi, np.pi, np.pi]),
            dtype="float64",
        )

        self.check_all_systems_ready()

    def call_service(self, service_name: str, service_class: Type[object], *call_args):
        rospy.wait_for_service(service_name)
        try:
            client = rospy.ServiceProxy(service_name, service_class)
            response = client(*call_args)
            assert response.result >= NR_E_NORMAL, f"{service_name} return error code: {response.rersult}"
            return response
        except rospy.ServiceException:
            rospy.logwarn(f"Service call failed: {service_name}")
            return None

    def update_robot_state(self):
        now = rospy.Time.now()
        self.tf_listener.waitForTransform(BASE_LINK_NAME, TOOL_LINK_NAME, now, rospy.Duration(2.0))
        (trans, quat) = self.tf_listener(BASE_LINK_NAME, TOOL_LINK_NAME, now)
        self.robot_state[:3] = np.array(trans, dtype=np.float64)  # m

        # rosとmujocoでクォータニオンの形式が異なるので変換
        m_quat = [quat[3]]
        m_quat += quat[:3]

        euler = rot.quat2euler(m_quat)  # rad
        self.robot_state[3:] = np.array(euler, dtype=np.float64)

    def rgb_image_callback(self, data: Image):
        try:
            bgr_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            self.update_display(bgr_image=bgr_image)
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

    def update_display(self, bgr_image: np.ndarray = None):
        if bgr_image is None:
            bgr_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_RGB2BGR)
        depth_image = cv2.cvtColor(self.depth_image, cv2.COLOR_GRAY2BGR)
        merged_image = np.vstack((bgr_image, depth_image))
        cv2.imshow("Images", merged_image)
        cv2.waitKey(1)

    def check_all_systems_ready(self):
        rospy.logdebug("Waiting for all systems to be ready...")
        self.check_all_sensors_ready()
        self.check_publishers_connection()
        rospy.loginfo("All Systems Ready")

    def check_all_sensors_ready(self):
        rospy.logdebug("Waiting for all sensros to be ready...")
        self.check_rgb_image_ready()
        self.check_depth_image_ready()
        rospy.loginfo("All Sensors Ready")

    def check_rgb_image_ready(self):
        rospy.logdebug(f"Waiting for {RGB_IMAGE_TOPIC_NAME} to be ready...")
        data = rospy.wait_for_message(RGB_IMAGE_TOPIC_NAME, Image)
        self.rgb_image_callback(data)
        rospy.logdebug(f"{RGB_IMAGE_TOPIC_NAME} Ready")

    def check_depth_image_ready(self):
        rospy.logdebug(f"Waiting for {DEPTH_IMAGE_TOPIC_NAME} to be ready...")
        data = rospy.wait_for_message(DEPTH_IMAGE_TOPIC_NAME, Image)
        self.depth_image_callback(data)
        rospy.logdebug(f"{DEPTH_IMAGE_TOPIC_NAME} Ready")

    def check_publishers_connection(self):
        rospy.logdebug("Waiting for all publishers to be connected...")
        self.check_position_command_pub_connection()
        rospy.loginfo("All Publishers Ready")

    def check_position_command_pub_connection(self):
        rospy.logdebug("Waiting for position_command_pub to be connected...")
        while self.position_command_pub.get_num_connections() == 0 and not rospy.is_shutdown():
            self.rate.sleep()
        rospy.logdebug("position_command_pub Connected")

    def set_action(self, action: np.ndarray):
        action = np.clip(action.copy(), -1, 1)

        # unscale
        pos_ctrl, rot_ctrl = action[:3], action[3:]
        pos_ctrl *= 0.05  # m
        rot_ctrl *= np.deg2rad(10)  # rad

        # 目標の計算
        pos_cur, rot_cur = self.robot_state[:3], np.deg2rad(self.robot_state[3:])
        pos_target = pos_cur + pos_ctrl
        mat_target = rot.add_rot_mat(rot.euler2mat(rot_cur), rot.euler2mat(rot_ctrl))
        rot_target = np.rad2deg(rot.mat2euler(mat_target))  # deg
        target = np.concatenate(pos_target, rot_target)

        # 指令の送信
        msg = Float64MultiArray()
        msg.data = target
        self.position_command_pub.publish(msg)
        rospy.logdebug(f"Published target: {target}")

        # 指令の受付まで待機、一定時間で先に進む
        start = rospy.Time.now()
        while not self.is_moving() and (rospy.Time.now() - start) <= rospy.Duration(0.5):
            self.rate.sleep()

        # 動作の終了まで待機
        while self.is_moving():
            self.rate.sleep()

    def is_moving(self) -> bool:
        response = self.call_service(
            SRV_NAME_GET_ACSGENERAL_OUTPUT_SIGNAL, getGeneralSignal, MOVING_GENERAL_OUTPUT_NUMBER, 1
        )
        if isinstance(response, getGeneralSignalResponse):
            return response.signal_state[0] == 1
        else:
            # サービスを取得できない場合は動作中にする
            return True

    def close(self):
        # モータをオフにする
        self.call_service(SRV_NAME_CTRLMOTER_OFF, TriggerWithResultCode)

        # 接続終了
        self.call_service(SRV_NAME_CLOSE, TriggerWithResultCode)
