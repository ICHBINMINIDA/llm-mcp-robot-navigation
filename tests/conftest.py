"""Test setup: a minimal fake of the ROS 2 Python API.

rclpy and the message packages only exist inside a ROS 2 installation, so the
tests replace them with small stand-ins. Everything else (stop flag, motion
state, watchdog, metrics, JSON handling) runs for real.
"""

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class _Logger:
    def info(self, *_): pass
    def warn(self, *_): pass
    def error(self, *_): pass
    def debug(self, *_): pass


class _Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


class _Node:
    def __init__(self, name):
        self.name = name

    def create_publisher(self, *_):
        return _Publisher()

    def create_subscription(self, *_):
        return None

    def get_logger(self):
        return _Logger()

    def destroy_node(self):
        pass


class _Vector:
    x = y = z = 0.0


class _Twist:
    def __init__(self):
        self.linear, self.angular = _Vector(), _Vector()


def _install_fake_ros() -> None:
    rclpy = types.ModuleType("rclpy")
    rclpy.init = lambda *a, **k: None
    rclpy.spin = lambda *a, **k: None
    rclpy.shutdown = lambda *a, **k: None
    node = types.ModuleType("rclpy.node")
    node.Node = _Node
    rclpy.node = node
    geometry = types.ModuleType("geometry_msgs")
    geometry_msg = types.ModuleType("geometry_msgs.msg")
    geometry_msg.Twist = _Twist
    sensor = types.ModuleType("sensor_msgs")
    sensor_msg = types.ModuleType("sensor_msgs.msg")
    sensor_msg.Image = sensor_msg.CompressedImage = object
    sys.modules.update({
        "rclpy": rclpy, "rclpy.node": node,
        "geometry_msgs": geometry, "geometry_msgs.msg": geometry_msg,
        "sensor_msgs": sensor, "sensor_msgs.msg": sensor_msg,
    })


_install_fake_ros()


@pytest.fixture
def stop_flag(tmp_path, monkeypatch):
    """Point both common modules at a private stop-flag file."""
    import llm_common as lc
    import robot_common as rc

    flag = tmp_path / "stop.flag"
    monkeypatch.setattr(rc, "STOP_FLAG", str(flag))
    monkeypatch.setattr(lc, "STOP_FLAG", str(flag))
    return flag


@pytest.fixture
def robot(stop_flag):
    import robot_common as rc

    node = rc.RobotNode("test_node")
    yield node
    node.shutdown()
