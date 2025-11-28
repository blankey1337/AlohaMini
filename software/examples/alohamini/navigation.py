import math
import time
import numpy as np

class PIDController:
    def __init__(self, kp, ki, kd, output_limits=(None, None)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min_out, self.max_out = output_limits
        
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = None
        
    def update(self, error, current_time=None):
        if current_time is None:
            current_time = time.perf_counter()
            
        if self.last_time is None:
            self.last_time = current_time
            dt = 0.0
        else:
            dt = current_time - self.last_time
            self.last_time = current_time
            
        # P term
        p_term = self.kp * error
        
        # I term
        if dt > 0:
            self.integral += error * dt
            # Clamp integral if needed (anti-windup)
        i_term = self.ki * self.integral
        
        # D term
        d_term = 0.0
        if dt > 0:
            d_term = self.kd * (error - self.prev_error) / dt
            
        self.prev_error = error
        
        output = p_term + i_term + d_term
        
        if self.min_out is not None:
            output = max(self.min_out, output)
        if self.max_out is not None:
            output = min(self.max_out, output)
            
        return output
    
    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = None

class NavigationController:
    def __init__(self):
        # PID for Linear Velocity (Drive to distance)
        # Target distance is 0.
        self.linear_pid = PIDController(kp=1.0, ki=0.0, kd=0.1, output_limits=(-0.5, 0.5))
        
        # PID for Angular Velocity (Turn to heading)
        # Target heading difference is 0.
        self.angular_pid = PIDController(kp=2.0, ki=0.0, kd=0.1, output_limits=(-90.0, 90.0))
        
        self.target_pose = None # {x, y}
        self.tolerance_dist = 0.1 # meters
        self.tolerance_angle = 5.0 # degrees

    def set_target(self, x, y):
        self.target_pose = {"x": x, "y": y}
        self.linear_pid.reset()
        self.angular_pid.reset()
        print(f"[NAV] Target set: ({x}, {y})")

    def get_action(self, current_pose):
        """
        current_pose: {x, y, theta (degrees)}
        Returns: {x.vel, y.vel, theta.vel} (Action dict)
        """
        if self.target_pose is None:
            return {}
            
        tx, ty = self.target_pose["x"], self.target_pose["y"]
        cx, cy, cth = current_pose["x"], current_pose["y"], current_pose["theta"]
        
        # Calculate distance to target
        dx = tx - cx
        dy = ty - cy
        dist = math.sqrt(dx*dx + dy*dy)
        
        if dist < self.tolerance_dist:
            print("[NAV] Target reached.")
            self.target_pose = None
            return {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
            
        # Calculate heading to target
        target_heading = math.degrees(math.atan2(dy, dx))
        
        # Calculate heading error (shortest path)
        heading_error = target_heading - cth
        # Normalize to [-180, 180]
        while heading_error > 180: heading_error -= 360
        while heading_error < -180: heading_error += 360
        
        # Control Logic:
        # 1. Turn to face target
        # 2. Drive forward
        
        # Use PID
        ang_vel = self.angular_pid.update(heading_error)
        
        # If facing roughly target, drive
        lin_vel = 0.0
        if abs(heading_error) < 45: # Drive only if somewhat aligned
            # Use distance as error for linear PID? 
            # Or just proportional to dist?
            # We want to minimize distance (error = dist - 0 = dist)
            lin_vel = self.linear_pid.update(dist)
            
            # Simple approach: constant speed scaled by alignment
            # lin_vel = 0.2 * (1.0 - abs(heading_error)/45.0)
            
        return {
            "x.vel": lin_vel, 
            "y.vel": 0.0, # Non-holonomic drive (like a tank/car), no strafing for now
            "theta.vel": ang_vel
        }
