#!/usr/bin/env python3

import os
import rclpy
from rclpy.node import Node

import torch
import numpy as np

# ROS 메시지
from sensor_msgs.msg import Image
from visualization_msgs.msg import MarkerArray, Marker
from cv_bridge import CvBridge
import cv2

class YoloLidarFusionNode(Node):
    def __init__(self):
        super().__init__('yolo_lidar_fusion_node')

        # YOLOv5 모델 로드: best.pt 경로 지정
        model_path = '/home/park/clu_ws/src/lidar_cluster/lidar_cluster/best.pt'
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
        self.model.conf = 0.5
        self.model.iou = 0.45

        self.bridge = CvBridge()

        # 카메라 구독 (토픽: /camera/color/image_raw)
        self.camera_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.camera_callback,
            10
        )

        # 라이다 바운딩 박스(MarkerArray) 구독 (토픽: /cluster_boxes)
        self.lidar_box_sub = self.create_subscription(
            MarkerArray,
            '/cluster_boxes',
            self.lidar_box_callback,
            10
        )

        # 융합 결과 퍼블리시 (MarkerArray)
        self.fusion_pub = self.create_publisher(
            MarkerArray,
            '/fusion_markers',
            10
        )

        # 내부 저장
        self.current_image = None
        self.yolo_detections = []  # 각 2D 박스: (x1,y1,x2,y2,conf,class)
        self.lidar_boxes = []      # 각 3D 박스 정보

        self.get_logger().info("YoloLidarFusionNode initialized.")

    def camera_callback(self, msg):
        """카메라 콜백 - YOLOv5로 2D 박스 검출"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return

        results = self.model(cv_image)
        det = results.xyxy[0].cpu().numpy()  # [x1,y1,x2,y2,conf,class]

        self.current_image = cv_image
        self.yolo_detections = []
        for d in det:
            x1, y1, x2, y2, conf, cls = d
            self.yolo_detections.append((x1, y1, x2, y2, conf, cls))

        self.fuse()

    def lidar_box_callback(self, msg):
        """MarkerArray로부터 3D 박스 정보를 추출"""
        self.lidar_boxes = []
        for marker in msg.markers:
            center = (marker.pose.position.x,
                      marker.pose.position.y,
                      marker.pose.position.z)
            scale = (marker.scale.x,
                     marker.scale.y,
                     marker.scale.z)
            orientation = (marker.pose.orientation.x,
                           marker.pose.orientation.y,
                           marker.pose.orientation.z,
                           marker.pose.orientation.w)
            self.lidar_boxes.append({
                'center': center,
                'scale': scale,
                'orientation': orientation,
                'id': marker.id
            })
        self.fuse()

    def fuse(self):
        """YOLO 2D 박스와 라이다 3D 박스를 융합 (투영 및 매칭)"""
        if self.current_image is None or not self.lidar_boxes:
            return

        fused_results = []
        for box3d in self.lidar_boxes:
            # project_3d_to_2d 함수가 None을 리턴하면 매칭 안 함
            projected_2d = self.project_3d_to_2d(box3d)
            if projected_2d is None:
                continue

            best_match = None
            best_iou = 0.0
            for det in self.yolo_detections:
                x1, y1, x2, y2, conf, cls = det
                iou = self.compute_iou(projected_2d, (x1, y1, x2, y2))
                if iou > best_iou:
                    best_iou = iou
                    best_match = det

            if best_match and best_iou > 0.1:
                fused_results.append({
                    'box3d': box3d,
                    'box2d': best_match,
                    'iou': best_iou
                })

        marker_array = MarkerArray()
        idx = 0
        for item in fused_results:
            box3d = item['box3d']
            center = box3d['center']
            scale = box3d['scale']

            marker = Marker()
            marker.header.frame_id = "velodyne"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "fusion"
            marker.id = idx
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = center[0]
            marker.pose.position.y = center[1]
            marker.pose.position.z = center[2]
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = scale[0]
            marker.scale.y = scale[1]
            marker.scale.z = scale[2]
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.5

            marker_array.markers.append(marker)
            idx += 1

        self.fusion_pub.publish(marker_array)
        self.get_logger().info(f"Fused {len(fused_results)} objects.")

    def project_3d_to_2d(self, box3d):
        """
        실제 구현 시: 라이다->카메라 캘리브레이션 행렬 및 카메라 Intrinsic 사용.
        현재 테스트용으로 임의의 2D 박스 값을 리턴.
        """
        # 임시: 모든 3D 박스에 대해 고정 2D 박스 (100, 100, 200, 200) 리턴
        return (100, 100, 200, 200)

    def compute_iou(self, boxA, boxB):
        """
        2D IoU 계산, boxA, boxB = (x1, y1, x2, y2)
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        if xB < xA or yB < yA:
            return 0.0
        interArea = (xB - xA) * (yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

def main(args=None):
    rclpy.init(args=args)
    node = YoloLidarFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node interrupted by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

