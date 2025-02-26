#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import open3d as o3d

# ROS 메시지
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray

class LidarClusteringNode(Node):
    def __init__(self):
        super().__init__('lidar_clustering_node')

        # (1) LiDAR 구독
        self.lidar_sub = self.create_subscription(
            PointCloud2,
            '/velodyne_points',  # 라이다 포인트 클라우드 토픽
            self.lidar_callback,
            10
        )

        # (2) 클러스터링 후 색상화된 PointCloud 퍼블리셔
        self.cluster_pub = self.create_publisher(
            PointCloud2,
            '/lidar_clusters',
            10
        )

        # (3) 각 클러스터 바운딩 박스를 MarkerArray로 퍼블리시
        self.bbox_pub = self.create_publisher(
            MarkerArray,
            '/cluster_boxes',
            10
        )

        self.get_logger().info("LidarClusteringNode started.")

    def lidar_callback(self, msg: PointCloud2):
        """ROS PointCloud2 -> Open3D -> (Z 필터) -> 노이즈 제거 -> 지면 제거(RANSAC) -> 클러스터링 -> 바운딩 박스 -> 퍼블리시"""

        # 1) PointCloud2 -> numpy
        points_list = []
        for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            x, y, z = p
            points_list.append([x, y, z])

        if not points_list:
            self.get_logger().warn("No valid points found.")
            return

        np_points = np.array(points_list, dtype=np.float32)

        # ----------------------------
        #   (A) Z 범위 필터 (나이브한 바닥/천장 제외)
        # ----------------------------
        z_min, z_max = -0.2, 2.0
        z_indices = np.where((np_points[:, 2] > z_min) & (np_points[:, 2] < z_max))[0]
        if len(z_indices) == 0:
            self.get_logger().warn("All points filtered out by Z range.")
            return
        np_points = np_points[z_indices]

        # 2) numpy -> Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np_points)

        # ----------------------------
        #   (B) 노이즈 제거 (통계적 아웃라이어)
        # ----------------------------
        pcd, inliers = pcd.remove_statistical_outlier(
            nb_neighbors=20,  
            std_ratio=1.0
        )

        # ----------------------------
        #   (C) 지면 제거 (RANSAC Plane Fitting)
        # ----------------------------
        # distance_threshold는 센서 높이/환경에 맞춰 조정
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.15,
            ransac_n=3,
            num_iterations=1000
        )
        # 바닥(inliers), 비바닥(outliers) 분리
        inlier_cloud = pcd.select_by_index(inliers)              # 지면
        outlier_cloud = pcd.select_by_index(inliers, invert=True)  # 지면 제외 (장애물)

        # 클러스터링에는 바닥을 뺀 outlier_cloud만 사용
        pcd = outlier_cloud

        # ----------------------------
        #   (D) DBSCAN 클러스터링
        # ----------------------------
        eps = 0.2       # 포인트 간 최대 거리
        min_points = 100 # 최소 포인트 수
        labels = np.array(pcd.cluster_dbscan(
            eps=eps,
            min_points=min_points,
            print_progress=False
        ))

        max_label = labels.max() if len(labels) > 0 else -1
        self.get_logger().info(f"Cluster labels: {labels}")

        # ----------------------------
        #   (E) 각 클러스터별 색상 부여
        # ----------------------------
        colors = self.colormap_labels(labels, max_label)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # (E-1) 클러스터가 색상화된 PointCloud2 퍼블리시
        self.publish_clustered_cloud(msg, pcd, labels)

        # ----------------------------
        #   (F) 바운딩 박스 MarkerArray 퍼블리시
        # ----------------------------
        self.publish_bounding_boxes(pcd, labels, msg.header)

    def publish_clustered_cloud(self, original_msg, pcd, labels):
        """Open3D pcd -> sensor_msgs/PointCloud2 퍼블리시 (intensity에 label 저장)"""
        final_points = np.asarray(pcd.points)
        header = original_msg.header
        header.stamp = self.get_clock().now().to_msg()

        fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        cloud_data = []
        for i in range(len(final_points)):
            x, y, z = final_points[i]
            label = labels[i]
            intensity = float(label) if label != -1 else 0.0
            cloud_data.append((x, y, z, intensity))

        cluster_cloud_msg = pc2.create_cloud(header, fields, cloud_data)
        self.cluster_pub.publish(cluster_cloud_msg)

    def publish_bounding_boxes(self, pcd, labels, header):
        """
        각 클러스터별 Axis-Aligned Bounding Box를 MarkerArray로 퍼블리시.
        노이즈(-1)는 스킵. 필요 시 '콘 크기 필터'도 추가 가능.
        """
        marker_array = MarkerArray()
        unique_labels = np.unique(labels)
        idx = 0

        for cluster_id in unique_labels:
            if cluster_id == -1:
                # 노이즈는 스킵
                continue

            indices = np.where(labels == cluster_id)[0]
            if len(indices) == 0:
                continue

            cluster_pcd = pcd.select_by_index(indices)
            aabb = cluster_pcd.get_axis_aligned_bounding_box()
            center = aabb.get_center()
            extent = aabb.get_extent()

            # (예시) 콘 높이 필터: 0.2m ~ 0.7m
            # if not (0.2 < extent[2] < 0.7):
            #     continue

            marker = Marker()
            marker.header.frame_id = header.frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "cluster_boxes"
            marker.id = idx
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = float(center[0])
            marker.pose.position.y = float(center[1])
            marker.pose.position.z = float(center[2])

            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0

            marker.scale.x = float(extent[0])
            marker.scale.y = float(extent[1])
            marker.scale.z = float(extent[2])

            # 노란색 반투명
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.5

            marker.lifetime.sec = 0
            marker_array.markers.append(marker)
            idx += 1

        self.bbox_pub.publish(marker_array)

    def colormap_labels(self, labels, max_label):
        """
        각 클러스터 레이블에 대해 무작위 색상 할당, 노이즈(-1)는 회색 처리
        """
        import matplotlib.pyplot as plt
        colors = np.zeros((labels.shape[0], 3))
        for cluster_id in range(max_label + 1):
            color = plt.cm.get_cmap("hsv", max_label+1)(cluster_id)[:3]
            indices = np.where(labels == cluster_id)[0]
            colors[indices] = color

        noise_indices = np.where(labels == -1)[0]
        colors[noise_indices] = [0.5, 0.5, 0.5]  # gray
        return colors

def main(args=None):
    rclpy.init(args=args)
    node = LidarClusteringNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node interrupted by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

