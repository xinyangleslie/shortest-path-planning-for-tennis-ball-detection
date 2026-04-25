import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray

class TestMarkerNode(Node):
    def __init__(self):
        super().__init__("test_marker_node")
        self.pub = self.create_publisher(MarkerArray, "/tennis_markers", 10)
        self.timer = self.create_timer(0.5, self.publish_marker)

    def publish_marker(self):
        arr = MarkerArray()

        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "test_ball"
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = 1.0
        m.pose.position.y = 0.0
        m.pose.position.z = 0.5
        m.pose.orientation.w = 1.0
        m.scale.x = 0.08
        m.scale.y = 0.08
        m.scale.z = 0.08
        m.color.a = 1.0
        m.color.r = 1.0
        m.color.g = 0.5
        m.color.b = 0.0

        arr.markers.append(m)
        self.pub.publish(arr)

def main():
    rclpy.init()
    node = TestMarkerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()