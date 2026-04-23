#!/usr/bin/env python3
from typing import Optional, Dict, List
from argparse import ArgumentParser
from math import sqrt, atan2, pi, inf
import math
import json
import numpy as np

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion

# Import your existing implementations
from lab8_9_starter import Map, ParticleFilter, angle_to_neg_pi_to_pi  # :contentReference[oaicite:2]{index=2}
from lab10_starter import RrtPlanner, PIDController as WaypointPID, GOAL_THRESHOLD  # :contentReference[oaicite:3]{index=3}


class PFRRTController:
    """
    Combined controller that:
      1) Localizes using a particle filter (by exploring).
      2) Plans with RRT from PF estimate to goal.
      3) Follows that plan with a waypoint PID controller while
         continuing to run the particle filter.
    """

    def __init__(self, pf: ParticleFilter, planner: RrtPlanner, goal_position: Dict[str, float]):
        self._pf = pf
        self._planner = planner
        self.goal_position = goal_position

        # Robot state from odom / laser
        self.current_position: Optional[Dict[str, float]] = None
        self.last_odom: Optional[Dict[str, float]] = None
        self.laserscan: Optional[LaserScan] = None

        # Command publisher
        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        # Subscribers
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)
        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.laserscan_callback)

        # PID controllers for tracking waypoints (copied from your ObstacleFreeWaypointController)
        self.linear_pid = WaypointPID(0.3, 0.0, 0.1, 10, -0.22, 0.22)
        self.angular_pid = WaypointPID(0.5, 0.0, 0.2, 10, -2.84, 2.84)

        # Waypoint tracking state
        self.plan: Optional[List[Dict[str, float]]] = None
        self.current_wp_idx: int = 0

        self.rate = rospy.Rate(10)

        # Wait until we have initial odom + scan
        while (self.current_position is None or self.laserscan is None) and (not rospy.is_shutdown()):
            rospy.loginfo("Waiting for /odom and /scan...")
            rospy.sleep(0.1)

    # ----------------------------------------------------------------------
    # Basic callbacks
    # ----------------------------------------------------------------------
    def odom_callback(self, msg: Odometry):
        pose = msg.pose.pose
        orientation = pose.orientation
        _, _, theta = euler_from_quaternion(
            [orientation.x, orientation.y, orientation.z, orientation.w]
        )

        new_pose = {"x": pose.position.x, "y": pose.position.y, "theta": theta}

        # Use odom delta to propagate PF motion model
        if self.last_odom is not None:
            dx_world = new_pose["x"] - self.last_odom["x"]
            dy_world = new_pose["y"] - self.last_odom["y"]
            dtheta = angle_to_neg_pi_to_pi(new_pose["theta"] - self.last_odom["theta"])

            # convert world delta to robot frame of previous pose
            ct = math.cos(self.last_odom["theta"])
            st = math.sin(self.last_odom["theta"])
            dx_robot = ct * dx_world + st * dy_world
            dy_robot = -st * dx_world + ct * dy_world

            # propagate all particles
            self._pf.move_by(dx_robot, dy_robot, dtheta)

        self.last_odom = new_pose
        self.current_position = new_pose

    def laserscan_callback(self, msg: LaserScan):
        self.laserscan = msg

    # ----------------------------------------------------------------------
    # Low-level motion primitives
    # ----------------------------------------------------------------------
    def move_forward(self, distance: float):
        """
        Move the robot straight by a commanded distance (meters)
        using a constant velocity profile.
        """
        twist = Twist()
        speed = 0.15  # m/s
        twist.linear.x = speed if distance >= 0 else -speed

        duration = abs(distance) / speed if speed > 0 else 0.0
        start_time = rospy.Time.now().to_sec()
        r = rospy.Rate(10)

        while (rospy.Time.now().to_sec() - start_time) < duration and (not rospy.is_shutdown()):
            self.cmd_pub.publish(twist)
            r.sleep()

        # Stop
        twist.linear.x = 0.0
        self.cmd_pub.publish(twist)

    def rotate_in_place(self, angle: float):
        """
        Rotate robot by a relative angle (radians).
        """
        twist = Twist()
        angular_speed = 0.8  # rad/s
        twist.angular.z = angular_speed if angle >= 0.0 else -angular_speed

        duration = abs(angle) / angular_speed if angular_speed > 0 else 0.0
        start_time = rospy.Time.now().to_sec()
        r = rospy.Rate(10)

        while (rospy.Time.now().to_sec() - start_time) < duration and (not rospy.is_shutdown()):
            self.cmd_pub.publish(twist)
            r.sleep()

        # Stop
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)

    # ----------------------------------------------------------------------
    # Measurement update
    # ----------------------------------------------------------------------
    def take_measurements(self):
        """
        Use 3 beams (-15°, 0°, +15° in the robot frame) from /scan
        to update the particle filter via its measurement model.
        """
        if self.laserscan is None:
            return

        angle_min = self.laserscan.angle_min
        angle_increment = self.laserscan.angle_increment
        ranges = self.laserscan.ranges
        num_ranges = len(ranges)

        mid_idx = num_ranges // 2
        offset = int(15.0 / (angle_increment * 180.0 / math.pi))  # 15 degrees offset

        indices = [max(0, min(num_ranges - 1, mid_idx + i)) for i in (-offset, 0, offset)]
        measurements = []

        for idx in indices:
            z = ranges[idx]
            if z == inf or np.isinf(z):
                if hasattr(self.laserscan, "range_max"):
                    z = self.laserscan.range_max
                else:
                    z = 10.0  # fallback
            angle = angle_min + idx * angle_increment  # angle in robot frame
            measurements.append((z, angle))

        for z, a in measurements:
            self._pf.measure(z, a)

    # ----------------------------------------------------------------------
    # Phase 1: Localization with PF (explore a bit)
    # ----------------------------------------------------------------------
    def localize_with_pf(self, max_steps: int = 400):
        """
        Simple autonomous exploration policy:
          - If front is free, go forward.
          - If obstacle close in front, back up and rotate.
        After each motion, apply PF measurement updates and check convergence.
        """

        ######### Your code starts here #########
        CONFIDENCE_THRESHOLD = 0.20
        MIN_STEPS = 20
        FRONT_OBS_THRESHOLD = 0.45
        FORWARD_STEP = 0.20

        for step in range(1, max_steps + 1):
            if rospy.is_shutdown():
                return

            # After each motion, apply PF measurement updates and check convergence.
            self.take_measurements()
            self._pf.visualize_particles()
            self._pf.visualize_estimate()

            try:
                xs = [p.x for p in self._pf._particles]
                ys = [p.y for p in self._pf._particles]
                x_std = float(np.std(xs))
                y_std = float(np.std(ys))
            except Exception:
                x_std, y_std = float("inf"), float("inf")

            if step >= MIN_STEPS and x_std < CONFIDENCE_THRESHOLD and y_std < CONFIDENCE_THRESHOLD:
                rospy.loginfo(f"PF converged (std x={x_std:.3f}, y={y_std:.3f})")
                return

            # Simple exploration policy based on front range
            scan = self.laserscan
            if scan is None or not scan.ranges:
                self.rate.sleep()
                continue

            idx0 = int(round((0.0 - scan.angle_min) / scan.angle_increment))
            idx0 = max(0, min(len(scan.ranges) - 1, idx0))
            front = scan.ranges[idx0]
            if front is None or np.isinf(front) or math.isnan(front) or front <= 0:
                front = float(getattr(scan, "range_max", 10.0))

            if float(front) < FRONT_OBS_THRESHOLD:
                self.move_forward(-0.10)
                self.rotate_in_place(pi / 2)
            else:
                self.move_forward(FORWARD_STEP)

            self.rate.sleep()

        rospy.logwarn("PF did not converge within max_steps; continuing anyway.")

        ######### Your code ends here #########

    # ----------------------------------------------------------------------
    # Phase 2: Planning with RRT
    # ----------------------------------------------------------------------
    def plan_with_rrt(self):
        """
        Generate a path using RRT from PF-estimated start to known goal.
        """
        ######### Your code starts here #########
        x, y, _ = self._pf.get_estimate()
        start = {"x": float(x), "y": float(y)}
        goal = {"x": float(self.goal_position["x"]), "y": float(self.goal_position["y"])}

        plan, graph = self._planner.generate_plan(start, goal)

        if plan is None or len(plan) == 0:
            raise RuntimeError(f"RRT could not find a plan from {start} to {goal}.")

        self.plan = plan
        self.current_wp_idx = 0

        if hasattr(self._planner, "visualize_plan"):
            self._planner.visualize_plan(plan)
        if hasattr(self._planner, "visualize_graph"):
            self._planner.visualize_graph(graph)

        ######### Your code ends here #########

    # ----------------------------------------------------------------------
    # Phase 3: Following the RRT path
    # ----------------------------------------------------------------------
    def follow_plan(self):
        """
        Follow the RRT waypoints using PID on (distance, heading) error.
        Keep updating PF along the way.
        """
        ######### Your code starts here #########
        if not self.plan:
            raise RuntimeError("No plan available. Call plan_with_rrt() first.")

        waypoint_tolerance = GOAL_THRESHOLD
        EMERGENCY_STOP_DIST = 0.35

        try:
            while not rospy.is_shutdown() and self.current_wp_idx < len(self.plan):
                self.take_measurements()
                self._pf.visualize_particles()
                self._pf.visualize_estimate()

                # Emergency stop if obstacle too close in front
                scan = self.laserscan
                if scan is not None:
                    rel = (0.0 - scan.angle_min) % (2 * pi)
                    idx0 = int(round(rel / scan.angle_increment))
                    idx0 = max(0, min(len(scan.ranges) - 1, idx0))
                    d0 = scan.ranges[idx0]
                    if not np.isinf(d0) and not math.isnan(d0) and d0 < EMERGENCY_STOP_DIST:
                        self.cmd_pub.publish(Twist())
                        self.rotate_in_place(pi / 6)
                        continue

                x, y, theta = self._pf.get_estimate()
                wp = self.plan[self.current_wp_idx]
                dx = float(wp["x"]) - float(x)
                dy = float(wp["y"]) - float(y)
                dist_err = sqrt(dx * dx + dy * dy)
                ang_err = angle_to_neg_pi_to_pi(atan2(dy, dx) - theta)

                if dist_err < waypoint_tolerance:
                    self.current_wp_idx += 1
                    self.cmd_pub.publish(Twist())
                    self.rate.sleep()
                    continue

                t = rospy.Time.now().to_sec()
                cmd = Twist()
                cmd.linear.x = float(self.linear_pid.control(dist_err, t))
                cmd.angular.z = float(self.angular_pid.control(ang_err, t))
                self.cmd_pub.publish(cmd)
                self.rate.sleep()
        finally:
            self.cmd_pub.publish(Twist())

        ######### Your code ends here #########

    # ----------------------------------------------------------------------
    # Top-level
    # ----------------------------------------------------------------------
    def run(self):
        self.localize_with_pf()
        self.plan_with_rrt()
        self.follow_plan()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--map_filepath", type=str, required=True)
    args = parser.parse_args()

    with open(args.map_filepath, "r") as f:
        map_data = json.load(f)
        obstacles = map_data["obstacles"]
        map_aabb = map_data["map_aabb"]
        if "goal_position" not in map_data:
            raise RuntimeError("Map JSON must contain a 'goal_position' field.")
        goal_position = map_data["goal_position"]

    # Initialize ROS node
    rospy.init_node("pf_rrt_combined", anonymous=True)

    # Build map + PF + RRT
    map_obj = Map(obstacles, map_aabb)
    num_particles = 240
    translation_variance = 0.003
    rotation_variance = 0.02
    measurement_variance = 0.25

    pf = ParticleFilter(
        map_obj,
        num_particles,
        translation_variance,
        rotation_variance,
        measurement_variance,
    )
    planner = RrtPlanner(obstacles, map_aabb)

    controller = PFRRTController(pf, planner, goal_position)

    try:
        controller.run()
    except rospy.ROSInterruptException:
        pass