#!/usr/bin/env python3
"""ROS2 node that subscribes to fisheye images, undistorts, and runs DPVO."""

import os
import signal
import sys

import cv2
import numpy as np
import torch
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from builtin_interfaces.msg import Time as RosTime

from dpvo.config import cfg
from dpvo.dpvo import DPVO
from dpvo import lietorch
from dpvo.lietorch import SE3
from dpvo.utils import Timer


# GeoScan B1 left camera calibration (Kalibr pinhole-equidistant), used as the
# built-in default when no `camchain` yaml is provided. Matches cam0 in
# ~/Documents/Datasets/geoscan/geoscan_camchain-imucam.yaml.
DEFAULT_INTRINSICS = [465.30269231092973, 464.44407907099975, 646.2240734070592, 486.9358630691829]
DEFAULT_DIST = [-0.0313464513145838, 0.03307070111936863, -0.03358616295118505, 0.010940367109062672]
DEFAULT_RESOLUTION = [1280, 1024]


class DPVONode(Node):
    def __init__(self):
        super().__init__('dpvo_node')
        self.get_logger().info('DPVO ROS2 node starting...')

        # --- Parameters (settable from launch or `--ros-args -p name:=value`) ---
        self.declare_parameter('network', 'dpvo.pth')
        self.declare_parameter('config', 'config/default.yaml')
        self.declare_parameter('camchain', '')  # Kalibr camchain yaml (cam0); empty = built-in GeoScan default
        self.declare_parameter('stride', 1)
        self.declare_parameter('save_trajectory', 'results/dpvo_ros_geoscan_traj.txt')
        self.declare_parameter('backend_thresh', 64.0)
        self.declare_parameter('image_topic', '/left_camera/image')
        self.declare_parameter('pose_topic', '/dpvo/pose')
        self.declare_parameter('path_topic', '/dpvo/path')
        self.declare_parameter('pose_frame_id', 'dpvo_world')
        self.declare_parameter('publish_pose', True)

        gp = lambda n: self.get_parameter(n).value
        self.network_path = gp('network')
        config_path = gp('config')
        self.stride = gp('stride')
        self.save_path = gp('save_trajectory')
        backend_thresh = gp('backend_thresh')
        image_topic = gp('image_topic')
        pose_topic = gp('pose_topic')
        path_topic = gp('path_topic')
        self.pose_frame_id = gp('pose_frame_id')
        self.publish_pose = gp('publish_pose')

        # Camera intrinsics + fisheye undistortion maps (from Kalibr yaml or default)
        self._setup_camera(gp('camchain'))

        # Load DPVO config (must happen before lazy DPVO() init on first frame)
        cfg.merge_from_file(config_path)
        cfg.BACKEND_THRESH = backend_thresh
        self.get_logger().info(
            f'network={self.network_path} config={config_path} '
            f'stride={self.stride} BACKEND_THRESH={backend_thresh}')

        # DPVO setup
        self.slam = None
        self.frame_count = 0
        self.timestamps = []
        self.saved = False

        # Pose publisher for glim_ext dpvo_frontend consumption.
        # Publishes per-frame PoseStamped after each DPVO processing call; the
        # pose is the latest estimate (w2c inverted to c2w) stamped with the
        # original image header timestamp so GLIM can time-sync precisely.
        if self.publish_pose:
            self.pose_pub = self.create_publisher(PoseStamped, pose_topic, 50)
            # Path accumulates every published pose so rviz can draw the live
            # trajectory as a line (PoseStamped alone only shows the current pose).
            self.path_pub = self.create_publisher(Path, path_topic, 10)
            self.path_msg = Path()
            self.path_msg.header.frame_id = self.pose_frame_id
            self.get_logger().info(f'Publishing poses to {pose_topic}, path to {path_topic}')

        # Subscribe to left camera (BEST_EFFORT to match sensor/replayer QoS)
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5000,
        )
        self.sub = self.create_subscription(Image, image_topic, self.image_cb, qos)
        self.get_logger().info(f'Subscribed to {image_topic} (BEST_EFFORT), waiting for images...')

    def _setup_camera(self, camchain):
        """Load cam0 intrinsics/distortion/resolution from a Kalibr camchain yaml
        (single source of truth) or fall back to the built-in GeoScan default,
        then precompute the fisheye-equidistant undistortion maps.
        """
        fx, fy, cx, cy = DEFAULT_INTRINSICS
        dist = DEFAULT_DIST
        width, height = DEFAULT_RESOLUTION
        if camchain:
            with open(camchain) as f:
                cam = yaml.safe_load(f)['cam0']
            fx, fy, cx, cy = cam['intrinsics']
            dist = cam['distortion_coeffs']
            width, height = cam['resolution']
            self.get_logger().info(f'Loaded cam0 intrinsics from {camchain}')
        else:
            self.get_logger().info('Using built-in GeoScan default intrinsics')

        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        D = np.array(dist)
        # Keep K unchanged for the undistorted output so INTRINSICS stays valid.
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), K, (int(width), int(height)), cv2.CV_16SC2)
        self.intrinsics = np.array([fx, fy, cx, cy])
        self.get_logger().info(
            f'Camera fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}, size {int(width)}x{int(height)}')

    @torch.no_grad()
    def image_cb(self, msg):
        self.frame_count += 1
        if self.frame_count % self.stride != 0:
            return

        # Convert ROS Image to numpy
        h, w = msg.height, msg.width
        encoding = msg.encoding
        if encoding in ('mono8', '8UC1'):
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif encoding in ('bgr8', '8UC3'):
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
        elif encoding == 'rgb8':
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            self.get_logger().warn(f'Unknown encoding: {encoding}, skipping')
            return

        # Undistort fisheye
        img_undist = cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)

        # Crop to multiple of 16
        h2, w2 = img_undist.shape[:2]
        img_undist = img_undist[:h2 - h2 % 16, :w2 - w2 % 16]

        # To tensor
        image_t = torch.from_numpy(img_undist).permute(2, 0, 1).cuda()
        intrinsics_t = torch.from_numpy(self.intrinsics).cuda()

        # Timestamp
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.timestamps.append(ts)
        t = len(self.timestamps) - 1

        # Init DPVO on first frame
        if self.slam is None:
            self.slam = DPVO(cfg, self.network_path, ht=image_t.shape[1], wd=image_t.shape[2], viz=False)
            self.get_logger().info(f'DPVO initialized, image size: {image_t.shape[1]}x{image_t.shape[2]}')

        with Timer("SLAM", enabled=False):
            self.slam(t, image_t, intrinsics_t)

        # Publish latest pose estimate for this frame to /dpvo/pose
        if self.publish_pose and self.slam.n > 0:
            self._publish_latest_pose(msg.header.stamp)

        if t % 100 == 0:
            self.get_logger().info(f'Processed frame {t}, total received: {self.frame_count}')

    def _publish_latest_pose(self, stamp):
        """Read the most recent pose from DPVO's sliding-window buffer, invert
        (w2c → c2w to match TUM convention), and publish as PoseStamped.
        """
        try:
            # pg.poses_ is (BUFFER_SIZE, 7) in world-to-camera SE3. Take the
            # latest valid entry.
            pose_w2c = self.slam.pg.poses_[self.slam.n - 1:self.slam.n].clone()  # (1,7)
            pose_c2w = SE3(pose_w2c).inv().data.cpu().numpy()[0]  # (7,)
        except Exception as e:
            self.get_logger().warn(f'pose extraction failed: {e}')
            return
        tx, ty, tz, qx, qy, qz, qw = pose_c2w

        ps = PoseStamped()
        ps.header.stamp = stamp  # preserve original image header stamp
        ps.header.frame_id = self.pose_frame_id
        ps.pose.position.x = float(tx)
        ps.pose.position.y = float(ty)
        ps.pose.position.z = float(tz)
        ps.pose.orientation.x = float(qx)
        ps.pose.orientation.y = float(qy)
        ps.pose.orientation.z = float(qz)
        ps.pose.orientation.w = float(qw)
        self.pose_pub.publish(ps)

        # Append to the growing Path and republish for the live rviz trajectory.
        # (This is the raw per-frame latest pose; DPVO still back-optimizes older
        # keyframes internally — the final saved TUM file is the refined version.)
        self.path_msg.header.stamp = stamp
        self.path_msg.poses.append(ps)
        self.path_pub.publish(self.path_msg)

    def _export_online_poses(self):
        """Build the trajectory from the current keyframe buffer, skipping the
        expensive final global BA. Mirrors DPVO.terminate()'s pose-interpolation
        tail (dpvo/dpvo.py) so the output shape matches (poses, tstamps).
        """
        s = self.slam
        s.traj = {}
        for i in range(s.n):
            s.traj[s.pg.tstamps_[i]] = s.pg.poses_[i]
        poses = [s.get_pose(t) for t in range(s.counter)]
        poses = lietorch.stack(poses, dim=0)
        poses = poses.inv().data.cpu().numpy()
        tstamps = np.array(s.tlist, dtype=np.float64)
        return poses, tstamps

    def save_trajectory(self):
        if self.saved:
            return
        if self.slam is None:
            self.get_logger().warn('No SLAM data to save')
            return

        self.get_logger().info('Terminating DPVO and saving trajectory...')
        torch.cuda.empty_cache()
        try:
            traj_est, tstamps_idx = self.slam.terminate()
        except torch.cuda.OutOfMemoryError:
            # terminate() runs 12 global-BA iterations over ALL keyframes, which
            # OOMs on small GPUs (e.g. 4060 8GB) for long sequences. Fall back to
            # the online sliding-window poses (no final global refinement).
            self.get_logger().warn(
                'terminate() OOM — skipping final global BA, exporting online VO poses.')
            torch.cuda.empty_cache()
            traj_est, tstamps_idx = self._export_online_poses()

        save_dir = os.path.dirname(os.path.abspath(self.save_path))
        os.makedirs(save_dir, exist_ok=True)
        with open(self.save_path, 'w') as f:
            f.write('# TUM format: timestamp tx ty tz qx qy qz qw\n')
            for i in range(len(tstamps_idx)):
                idx = int(tstamps_idx[i])
                if idx < len(self.timestamps):
                    ts = self.timestamps[idx]
                else:
                    ts = float(idx)
                tx, ty, tz = traj_est[i, :3]
                qx, qy, qz, qw = traj_est[i, 3:7]
                f.write(f"{ts:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")

        self.saved = True
        self.get_logger().info(f'Saved {len(tstamps_idx)} poses to {self.save_path}')

    def shutdown_and_exit(self):
        """Save, release CUDA, and force-exit.

        After a long sequence DPVO's process sometimes lingers holding its CUDA
        context (~GB of VRAM), which blocks the next launch (import-time OOM).
        We release what we can and then os._exit() to guarantee the OS reclaims
        the GPU memory — bypassing any hung non-daemon thread or stuck CUDA
        context that a clean rclpy shutdown would wait on.
        """
        try:
            self.save_trajectory()
        except Exception as e:
            self.get_logger().warn(f'save on shutdown failed: {e}')
        try:
            self.slam = None
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception:
            pass
        self.get_logger().info('DPVO node exiting, VRAM released.')
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


def main(args=None):
    torch.manual_seed(1234)

    rclpy.init(args=args)
    node = DPVONode()

    # Handle Ctrl+C gracefully (save + release VRAM + hard exit)
    def shutdown_handler(sig, frame):
        node.shutdown_and_exit()

    signal.signal(signal.SIGINT, shutdown_handler)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Normal spin return or SIGINT: save trajectory + release VRAM on the
        # way out. The node never self-terminates on idle -- it runs until the
        # process is stopped (Ctrl-C / SIGINT), which is correct for a live
        # streaming node fed by a camera or an externally-controlled bag.
        node.shutdown_and_exit()


if __name__ == '__main__':
    main()
