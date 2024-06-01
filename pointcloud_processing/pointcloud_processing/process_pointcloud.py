import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import numpy as np
import struct

class PointCloudProcessor(Node):
    def __init__(self):
        super().__init__('pointcloud_processor')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/converted_pointcloud2',
            self.listener_callback,
            10)
        self.publisher = self.create_publisher(PointCloud2, '/euclidean_distances', 10)

    def listener_callback(self, msg):
        points = self.convert_pointcloud2_to_xyz(msg)
        distances = np.linalg.norm(points, axis=1)
        self.get_logger().info(f'Publishing Euclidean distances')
        distance_cloud = self.create_distance_pointcloud(msg.header, distances)
        self.publisher.publish(distance_cloud)

    def convert_pointcloud2_to_xyz(self, cloud_msg):
        assert isinstance(cloud_msg, PointCloud2)
        points_list = []
        for point in self.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True):
            points_list.append([point[0], point[1], point[2]])
        return np.array(points_list)

    def read_points(self, cloud, field_names=None, skip_nans=False, uvs=[]):
        fmt = self._get_struct_fmt(cloud, field_names)
        unpack_from = struct.Struct(fmt).unpack_from
        point_step = cloud.point_step
        row_step = cloud.row_step
        data = cloud.data
        for u, v in uvs or self._get_default_uvs(cloud):
            offset = (v * row_step) + (u * point_step)
            point = unpack_from(data, offset)
            if skip_nans and any(np.isnan(p) for p in point):
                continue
            yield point

    def _get_struct_fmt(self, cloud, field_names):
        fmt = '>' if cloud.is_bigendian else '<'
        for field in cloud.fields:
            if field_names and field.name not in field_names:
                continue
            fmt += {
                1: 'B',  # uint8
                2: 'H',  # uint16
                4: 'f',  # float32
                8: 'd'   # float64
            }[field.count * 1]
        return fmt

    def _get_default_uvs(self, cloud):
        width, height = cloud.width, cloud.height
        return [(u, v) for v in range(height) for u in range(width)]

    def create_distance_pointcloud(self, header, distances):
        fields = [
            PointField(name='distance', offset=0, datatype=PointField.FLOAT32, count=1)
        ]
        points = np.array(distances, dtype=np.float32).tobytes()
        distance_cloud = PointCloud2(
            header=header,
            height=1,
            width=len(distances),
            fields=fields,
            is_bigendian=False,
            point_step=4,
            row_step=4 * len(distances),
            data=points,
            is_dense=True
        )
        return distance_cloud

def main(args=None):
    rclpy.init(args=args)
    pointcloud_processor = PointCloudProcessor()
    rclpy.spin(pointcloud_processor)
    pointcloud_processor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

