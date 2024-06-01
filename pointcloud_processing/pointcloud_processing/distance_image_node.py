import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from sensor_msgs_py import point_cloud2
import numpy as np
import cv2
from cv_bridge import CvBridge

class PointCloudToDistanceImage(Node):
    def __init__(self):
        super().__init__('pointcloud_to_distance_image')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/converted_pointcloud2',
            self.pointcloud_callback,
            10)
        self.publisher = self.create_publisher(Image, '/distance_image', 10)
        self.bridge = CvBridge()
        self.image_height = 480
        self.image_width = 640
        self.theta_min = -np.pi / 2  # 正面の最小角度
        self.theta_max = np.pi / 2   # 正面の最大角度

    def pointcloud_callback(self, msg):
        points = np.array(list(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))
        distance_image, label_image = self.create_distance_image(points)
        labeled_distance_image = self.label_distance_image(distance_image, label_image)
        colored_distance_image = self.apply_colormap(labeled_distance_image)
        image_msg = self.bridge.cv2_to_imgmsg(colored_distance_image, encoding="bgr8")
        self.publisher.publish(image_msg)

    def create_distance_image(self, points):
        distance_image = np.full((self.image_height, self.image_width), np.nan, dtype=np.float32)
        label_image = np.zeros((self.image_height, self.image_width), dtype=np.int32)
        label_counter = 1

        max_r = 0

        for point in points:
            x, y, z = point
            if x > 0:
                r = np.sqrt(x**2 + y**2)
                if r > max_r:
                    max_r = r

        if max_r == 0:
            self.get_logger().warn('Max distance r is zero. No valid points.')
            return distance_image, label_image  # Return empty images if max_r is zero

        for point in points:
            x, y, z = point
            if x > 0:  # Only consider points in front of the sensor
                r = np.sqrt(x**2 + y**2)
                theta = np.arctan2(y, x)

                if self.theta_min <= theta <= self.theta_max:
                    theta_norm = (theta - self.theta_min) / (self.theta_max - self.theta_min)
                    u = int(theta_norm * self.image_width)
                    v = int((z / max_r) * self.image_height)

                    if 0 <= u < self.image_width and 0 <= v < self.image_height:
                        distance_image[v, u] = r
                        if label_image[v, u] == 0:
                            neighbors = label_image[max(0, v-1):min(self.image_height, v+2), max(0, u-1):min(self.image_width, u+2)]
                            min_label = np.min(neighbors[neighbors > 0]) if np.any(neighbors > 0) else 0
                            if min_label == 0:
                                label_image[v, u] = label_counter
                                label_counter += 1
                            else:
                                label_image[v, u] = min_label

        return distance_image, label_image

    def label_distance_image(self, distance_image, label_image):
        unique_labels = np.unique(label_image)
        if unique_labels[0] == 0:
            unique_labels = unique_labels[1:]

        label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels, start=1)}

        labeled_image = np.copy(distance_image)

        for v in range(self.image_height):
            for u in range(self.image_width):
                if label_image[v, u] > 0:
                    labeled_image[v, u] = label_map[label_image[v, u]]

        return labeled_image

    def apply_colormap(self, labeled_image):
        # Normalize the distance image to the range 0-255
        norm_image = cv2.normalize(labeled_image, None, 0, 255, cv2.NORM_MINMAX)
        norm_image = np.nan_to_num(norm_image)  # Convert NaNs to 0
        norm_image = norm_image.astype(np.uint8)

        # Apply the colormap
        colored_image = cv2.applyColorMap(norm_image, cv2.COLORMAP_JET)

        return colored_image

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudToDistanceImage()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

