"""A minimal visual driving task: curving road, obstacles, a rendered camera image.

Deliberately not CARLA. Everything here is numpy so it runs on the Mac, and the
question it asks is the one CARLA cannot answer for us: given pixels, can the
connectome-derived controller keep a car on a road and off the obstacles.

Units are metres and radians. The car uses a kinematic bicycle model at fixed
speed; the camera is a pinhole looking at a flat ground plane.
"""
import os
import numpy as np

LANE_HALF_WIDTH = 3.5
WHEELBASE = 2.6
SPEED = 8.0
DT = 0.05                 # 20 Hz, the same control period as the wheel task
MAX_STEER = 0.5
STEER_RATE = 4.0          # rad/s slew limit on the commanded angle
CAM_HEIGHT = 1.25
IMAGE_W, IMAGE_H = 48, 24
FOCAL = 26.0
OBSTACLE_RADIUS = 1.0
OBSTACLE_HEIGHT = 1.6     # v2: obstacles are upright cylinders, not ground discs
TASK_VERSION = int(os.environ.get('FLY_TASK_VERSION', '2'))   # 1 reproduces the disc renderer


class Road:
    """Centreline y = c(x), built from a few bounded sinusoids, plus obstacles."""

    def __init__(self, seed, obstacles=True, length=900.0):
        rng = np.random.default_rng(seed)
        self.amps = rng.uniform(1.5, 4.5, 3) * rng.choice([-1.0, 1.0], 3)
        self.freqs = rng.uniform(0.018, 0.048, 3)
        self.phases = rng.uniform(0, 2*np.pi, 3)
        self.length = length
        if obstacles:
            xs = np.arange(45.0, length-40.0, rng.uniform(38.0, 52.0))
            xs = xs + rng.uniform(-6, 6, len(xs))
            side = rng.choice([-1.0, 1.0], len(xs))
            offs = side * rng.uniform(1.1, 2.0, len(xs))
            self.obstacles = np.stack([xs, self.centre(xs) + offs], axis=1)
        else:
            self.obstacles = np.zeros((0, 2))

    def centre(self, x):
        x = np.asarray(x, dtype=float)
        return sum(a*np.sin(f*x + p) for a, f, p in zip(self.amps, self.freqs, self.phases))

    def heading(self, x):
        x = np.asarray(x, dtype=float)
        slope = sum(a*f*np.cos(f*x + p) for a, f, p in zip(self.amps, self.freqs, self.phases))
        return np.arctan(slope)

    def offset(self, x, y):
        """Signed lateral offset from the centreline, corrected for road slope."""
        return (np.asarray(y) - self.centre(x)) * np.cos(self.heading(x))


class Car:
    def __init__(self, road, rng):
        self.road = road
        self.x = 5.0
        self.y = road.centre(self.x) + rng.uniform(-1.0, 1.0)
        self.theta = road.heading(self.x) + rng.uniform(-0.05, 0.05)
        self.steer = 0.0

    def step(self, command):
        target = float(np.clip(command, -MAX_STEER, MAX_STEER))
        delta = STEER_RATE*DT
        self.steer += np.clip(target - self.steer, -delta, delta)
        self.theta += SPEED/WHEELBASE*np.tan(self.steer)*DT
        self.x += SPEED*np.cos(self.theta)*DT
        self.y += SPEED*np.sin(self.theta)*DT

    @property
    def lateral(self):
        return float(self.road.offset(self.x, self.y))

    def crashed(self):
        if abs(self.lateral) > LANE_HALF_WIDTH:
            return 'off_road'
        if len(self.road.obstacles):
            d = np.hypot(self.road.obstacles[:, 0]-self.x, self.road.obstacles[:, 1]-self.y)
            if d.min() < OBSTACLE_RADIUS:
                return 'obstacle'
        return None


_U, _V = np.meshgrid(np.arange(IMAGE_W), np.arange(IMAGE_H))
_DIR = np.stack([np.ones_like(_U, dtype=float),
                 (IMAGE_W/2 - 0.5 - _U)/FOCAL,
                 (IMAGE_H/2 - 0.5 - _V)/FOCAL - 0.22], axis=-1)


def render(car):
    """Ray-cast the flat ground plane. Road is bright, verge dark, obstacles black."""
    dz = _DIR[..., 2]
    valid = dz < -1e-6
    t = np.where(valid, CAM_HEIGHT/np.where(valid, -dz, 1.0), 0.0)
    fx, fy = t*_DIR[..., 0], t*_DIR[..., 1]
    c, s = np.cos(car.theta), np.sin(car.theta)
    wx = car.x + c*fx - s*fy
    wy = car.y + s*fx + c*fy
    image = np.full((IMAGE_H, IMAGE_W), 0.15)
    on_road = valid & (np.abs(car.road.offset(wx, wy)) < LANE_HALF_WIDTH) & (t < 90.0)
    image[on_road] = 0.85
    # v1 drew obstacles as discs on the ground, which a 24-row image cannot
    # resolve beyond ~8 m: the rows near the horizon are metres apart on the
    # ground, so a 1 m disc fell between them. v2 intersects each ray with an
    # upright cylinder, so an obstacle 12 m out is a few rows tall and visible.
    hx = c*_DIR[..., 0] - s*_DIR[..., 1]
    hy = s*_DIR[..., 0] + c*_DIR[..., 1]
    fz = _DIR[..., 2]
    ground_t = np.where(valid, t, np.inf)
    for ox, oy in car.road.obstacles:
        if abs(ox - car.x) < 70 and TASK_VERSION == 1:
            image[valid & (np.hypot(wx-ox, wy-oy) < OBSTACLE_RADIUS)] = 0.0
        elif abs(ox - car.x) < 70:
            rx, ry = car.x - ox, car.y - oy
            a = hx*hx + hy*hy
            b = 2*(rx*hx + ry*hy)
            cc = rx*rx + ry*ry - OBSTACLE_RADIUS**2
            disc = b*b - 4*a*cc
            root = np.sqrt(np.maximum(disc, 0.0))
            tc = (-b - root)/(2*a)
            z = CAM_HEIGHT + tc*fz
            hit = (disc > 0) & (tc > 0) & (tc < ground_t) & (z > 0) & (z < OBSTACLE_HEIGHT)
            image[hit] = 0.0
            t = np.where(hit, tc, t)
    # Distance haze, so far pixels carry less weight than near ones.
    haze = np.clip(1.0 - t/110.0, 0.0, 1.0)
    return (image*haze + 0.15*(1-haze)).astype(np.float32)


def expert(car, lookahead=11.0):
    """Obstacle-aware pure pursuit. Used only to produce demonstrations."""
    ax = car.x + lookahead
    target_offset = 0.0
    if len(car.road.obstacles):
        ahead = car.road.obstacles[(car.road.obstacles[:, 0] > car.x + 2) &
                                   (car.road.obstacles[:, 0] < car.x + 26)]
        if len(ahead):
            ox, oy = ahead[np.argmin(ahead[:, 0])]
            o_off = float(car.road.offset(ox, oy))
            clear = OBSTACLE_RADIUS + 0.9
            if abs(o_off) < clear:
                left, right = o_off + clear, o_off - clear
                target_offset = left if abs(left) < abs(right) else right
                target_offset = float(np.clip(target_offset,
                                              -LANE_HALF_WIDTH+0.6, LANE_HALF_WIDTH-0.6))
            ax = min(ax, max(ox - 3.0, car.x + 5.0))
    ay = car.road.centre(ax) + target_offset/np.cos(car.road.heading(ax))
    dx, dy = ax - car.x, ay - car.y
    alpha = np.arctan2(dy, dx) - car.theta
    alpha = (alpha + np.pi) % (2*np.pi) - np.pi
    ld = np.hypot(dx, dy)
    return float(np.clip(np.arctan2(2*WHEELBASE*np.sin(alpha), ld), -MAX_STEER, MAX_STEER))


def rollout(seed, policy, steps=260, obstacles=True):
    """Run one closed-loop episode. `policy(image, car) -> steering command`."""
    road = Road(seed, obstacles=obstacles)
    car = Car(road, np.random.default_rng(seed + 7919))
    lateral, failure = [], None
    for _ in range(steps):
        car.step(policy(render(car), car))
        lateral.append(abs(car.lateral))
        failure = car.crashed()
        if failure:
            break
    return {'steps': len(lateral), 'failure': failure,
            'mean_abs_lateral': float(np.mean(lateral)),
            'max_abs_lateral': float(np.max(lateral)),
            'distance_m': float(car.x)}
