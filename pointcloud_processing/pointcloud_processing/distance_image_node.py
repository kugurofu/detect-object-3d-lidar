import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from sensor_msgs_py import point_cloud2
import numpy as np
import cv2
from cv_bridge import CvBridge

class PointCloudToDepthImage(Node):
    def __init__(self):
        super().__init__('pointcloud_to_depth_image')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/converted_pointcloud2',
            self.pointcloud_callback,
            10)
        self.publisher = self.create_publisher(Image, '/depth_image', 10)
        self.bridge = CvBridge()
        self.image_height = 480
        self.image_width = 640
        self.y_min = -np.pi / 2  # -90度（ラジアン）に相当
        self.y_max = np.pi / 2   # 90度（ラジアン）に相当
        self.z_min = -2.0   # 高さ方向（z軸）の最小値（調整可能）
        self.z_max = 2.0    # 高さ方向（z軸）の最大値（調整可能）

    def pointcloud_callback(self, msg):
        points = np.array(list(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))
        depth_image = self.create_depth_image(points)
        colored_depth_image = self.apply_colormap(depth_image)
        image_msg = self.bridge.cv2_to_imgmsg(colored_depth_image, encoding="bgr8")
        self.publisher.publish(image_msg)

    def create_depth_image(self, points):
        depth_image = np.full((self.image_height, self.image_width), np.nan, dtype=np.float32)

        for point in points:
            x, y, z = point

            # 前方のみを対象とするため、x > 0を確認
            if x > 0:
                theta = np.arctan2(y, x)  # y軸方向の角度を計算

                if self.y_min <= theta <= self.y_max and self.z_min <= z <= self.z_max:
                    y_norm = (theta - self.y_min) / (self.y_max - self.y_min)
                    z_norm = (z - self.z_min) / (self.z_max - self.z_min)
                    u = int(y_norm * self.image_width)
                    v = int((1 - z_norm) * self.image_height)  # 上から下に行くほどzが増えるようにする

                    if 0 <= u < self.image_width and 0 <= v < self.image_height:
                        depth_image[v, u] = x  # xの値を深度として使用

        return depth_image

    def apply_colormap(self, depth_image):
        valid_mask = ~np.isnan(depth_image)
        depths = depth_image[valid_mask]

        # カラーマップ用の空の画像を作成（白背景）
        colored_image = np.full((self.image_height, self.image_width, 3), 255, dtype=np.uint8)

        if len(depths) == 0:
            return colored_image  # 有効な深度値がなければ白い画像を返す

        min_depth = np.min(depths)
        max_depth = np.max(depths)

        norm_depths = ((depths - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)
        colormap = cv2.applyColorMap(norm_depths, cv2.COLORMAP_JET)

        # valid_maskを2次元から1次元に展開し、それに対応するcolormapを適用
        colored_image[valid_mask] = colormap.reshape(-1, 3)

        return colored_image

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudToDepthImage()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

