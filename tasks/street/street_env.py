"""A street-like 3D driving task, rendered by ray-casting in numpy.

Two-lane road with centre and edge markings, curbs and sidewalks, buildings of
varying height along both sides, lamp posts, parked cars on the shoulder that
intrude into the driving lane, oncoming traffic in the other lane and, from
task version 4, slow traffic in the driver's own lane that has to be overtaken
through the oncoming lane. The car must hold the right-hand lane, swerve inside
it around the parked cars, and pull out and back in around the slow cars.

The policy sees a shaded 64x32 first-person render of this scene; the video
renderer draws the same world from a chase camera behind the driver's car and
at higher resolution. Car dynamics, control period and the expert are the same
family as drive_env.py so results stay comparable.
"""
import os
import numpy as np

LANE_W = 3.5                    # one lane; the road is two lanes wide
ROAD_HALF = LANE_W               # centreline at 0, road from -3.5 to +3.5
LANE_CENTRE = -LANE_W/2          # -1.75: with y to the left of travel, the right-hand lane is negative
LEFT_LANE = LANE_W/2             # +1.75: the oncoming lane, used for overtaking
WHEELBASE = 2.6
SPEED = 8.0
DT = 0.05
MAX_STEER = 0.5
STEER_RATE = 4.0
CAM_HEIGHT = 1.35
IMAGE_W, IMAGE_H = 64, 32
FOCAL = 36.0
PITCH = -0.16
CAR_HALF_W, CAR_HALF_L = 0.9, 2.1
ONCOMING_SPEED = 8.0
TRAFFIC_SPEED = 3.0              # slow same-direction cars; the driver closes at 5 m/s
TASK_VERSION = int(os.environ.get('FLY_TASK_VERSION', 5))
PULL_OUT = 22.0 if TASK_VERSION < 5 else 16.0   # gap to the slow car at which the expert pulls out

# Materials, only used to colour the cinematic render; the policy sees grayscale.
SKY, GRASS, ROAD, MARKING, SIDEWALK, CURB, BUILDING, LAMP, PARKED, ONCOMING, TRAFFIC, OWN, FLY_BODY, FLY_HEAD, FLY_EYE, FLY_WING, FLY_SHADOW = range(17)
BOX, ELLIPSOID = 0, 1


class Road:
    """Centreline y = c(x) plus a scene: buildings, posts, parked, oncoming and slow cars."""

    def __init__(self, seed, obstacles=True, length=900.0):
        rng = np.random.default_rng(seed)
        self.amps = rng.uniform(1.5, 4.5, 3)*rng.choice([-1.0, 1.0], 3)
        self.freqs = rng.uniform(0.018, 0.048, 3)
        self.phases = rng.uniform(0, 2*np.pi, 3)
        self.length = length
        boxes = []       # static axis-aligned boxes: x0,x1,y0,y1,z0,z1 in ROAD coords (d = lateral), albedo
        for side in (-1.0, 1.0):
            x = 10.0
            while x < length:
                w = rng.uniform(8, 20); depth = rng.uniform(6, 14); h = rng.uniform(4, 18)
                gap = rng.uniform(0, 6)
                if rng.random() < 0.85:
                    d0 = side*6.0; d1 = side*(6.0+depth)
                    boxes.append([x, x+w, min(d0, d1), max(d0, d1), 0.0, h, rng.uniform(0.35, 0.8)])
                x += w + gap
        for x in np.arange(20.0, length, 25.0):
            for side in (-1.0, 1.0):
                boxes.append([x-0.12, x+0.12, side*5.0-0.12, side*5.0+0.12, 0.0, 5.0, 0.55])
        self.static = np.array(boxes, dtype=float)
        self.parked, self.oncoming, self.traffic = np.zeros((0, 2)), np.zeros((0, 2)), np.zeros((0, 2))
        if obstacles:
            xs = np.arange(45.0, length-40.0, rng.uniform(38.0, 52.0)) + rng.uniform(-6, 6)
            # Parked cars sit on the right shoulder and intrude ~0.8 m into the lane.
            self.parked = np.stack([xs, -rng.uniform(ROAD_HALF-1.1, ROAD_HALF-0.6, len(xs))], axis=1)
            # Oncoming cars start far ahead in the left lane; spaced so none passes a parked car
            # while the driver is around it (their meeting points are kept clear of parked-car zones).
            t0 = rng.uniform(1.0, 3.0)
            starts = []
            while t0 < 26.0:
                # The driver is near x = 5 + 8 t at time t; keep meeting points 20 m clear of parked cars.
                if np.all(np.abs(self.parked[:, 0] - (5.0 + SPEED*t0)) > 20.0):
                    starts.append(t0); t0 += rng.uniform(2.5, 5.0)
                else:
                    t0 += 0.5
            # Oncoming car i is at x = x_meet + ONCOMING_SPEED*(t_meet - t) so it meets the driver at t_meet.
            self.oncoming = np.array([[5.0 + SPEED*t, t] for t in starts]).reshape(-1, 2)   # (meet_x, meet_t)
            if TASK_VERSION >= 4:
                # One slow car in the driver's lane, caught at t_c (driver then near x = 5 + 8 t_c).
                # It is drawn after the v3 scene so version-3 roads are unchanged; parked cars near the
                # overtake and oncoming cars that would meet the driver while it is in the left lane
                # (from pull-out about 4.4 s before the catch to pull-in about 2.5 s after) are dropped.
                catch = rng.uniform(6.0, 11.0)
                self.traffic = np.array([[5.0 + (SPEED-TRAFFIC_SPEED)*catch, TRAFFIC_SPEED]])   # (x at t=0, speed)
                self.parked = self.parked[np.abs(self.parked[:, 0] - (5.0 + SPEED*catch)) > 30.0]
                # Denser oncoming traffic than v3: a meeting every 1.8-3.5 s except while the driver
                # is in the left lane or right beside a parked car.
                t0, starts = rng.uniform(1.0, 2.5), []
                while t0 < 26.0:
                    if np.all(np.abs(self.parked[:, 0] - (5.0 + SPEED*t0)) > 14.0) and (t0 < catch - 5.5 or t0 > catch + 3.5):
                        starts.append(t0); t0 += rng.uniform(1.8, 3.5)
                    else:
                        t0 += 0.4
                self.oncoming = np.array([[5.0 + SPEED*t, t] for t in starts]).reshape(-1, 2)

    def centre(self, x):
        x = np.asarray(x, dtype=float)
        return sum(a*np.sin(f*x + p) for a, f, p in zip(self.amps, self.freqs, self.phases))

    def heading(self, x):
        x = np.asarray(x, dtype=float)
        slope = sum(a*f*np.cos(f*x + p) for a, f, p in zip(self.amps, self.freqs, self.phases))
        return np.arctan(slope)

    def offset(self, x, y):
        return (np.asarray(y) - self.centre(x))*np.cos(self.heading(x))

    def oncoming_positions(self, t):
        """(x, lateral) of each oncoming car at time t, in road coordinates."""
        if not len(self.oncoming):
            return np.zeros((0, 2))
        x = self.oncoming[:, 0] + ONCOMING_SPEED*(self.oncoming[:, 1] - t)
        return np.stack([x, np.full(len(x), -LANE_CENTRE)], axis=1)   # +1.75, the oncoming lane

    def traffic_positions(self, t):
        """(x, lateral) of each slow same-direction car at time t. They hold the right lane and
        swerve inside it past parked cars the same way the expert does."""
        if not len(self.traffic):
            return np.zeros((0, 2))
        x = self.traffic[:, 0] + self.traffic[:, 1]*t
        lat = np.full(len(x), LANE_CENTRE)
        for px, pd in self.parked:
            w = np.clip(1.0 - (np.abs(x - px) - 5.0)/5.0, 0.0, 1.0)     # full swerve within 5 m, fading out by 10 m
            lat = np.maximum(lat, LANE_CENTRE + w*(_swerve_offset(pd) - LANE_CENTRE))
        return np.stack([x, lat], axis=1)


def _swerve_offset(pd):
    """Lateral target that clears a parked car at lateral pd while staying in the right lane."""
    return min(max(LANE_CENTRE, pd + 0.9 + CAR_HALF_W + 0.5), -0.2)


class Car:
    def __init__(self, road, rng):
        self.road = road
        self.x = 5.0
        self.y = road.centre(self.x) + LANE_CENTRE + rng.uniform(-0.6, 0.6)
        self.theta = road.heading(self.x) + rng.uniform(-0.05, 0.05)
        self.steer = 0.0
        self.t = 0.0

    def step(self, command):
        target = float(np.clip(command, -MAX_STEER, MAX_STEER))
        delta = STEER_RATE*DT
        self.steer += np.clip(target - self.steer, -delta, delta)
        self.theta += SPEED/WHEELBASE*np.tan(self.steer)*DT
        self.x += SPEED*np.cos(self.theta)*DT
        self.y += SPEED*np.sin(self.theta)*DT
        self.t += DT

    @property
    def lateral(self):
        return float(self.road.offset(self.x, self.y))

    def crashed(self):
        lat = self.lateral
        if lat > ROAD_HALF - 0.3 or lat < -ROAD_HALF + 0.3:
            return 'off_road'
        for kind, cars in (('parked_car', self.road.parked), ('oncoming_car', self.road.oncoming_positions(self.t)),
                           ('traffic_car', self.road.traffic_positions(self.t))):
            for cx, cd in cars:
                if abs(cx - self.x) < CAR_HALF_L + 2.1 and abs(cd - lat) < CAR_HALF_W + 0.9:
                    return kind
        return None


FAILURES = ('off_road', 'parked_car', 'oncoming_car', 'traffic_car')


def _car_boxes(rows, road, x, d, alb, mat, cabin_alb, forward=True):
    """A car as two oriented boxes: body and cabin, aligned with the road heading at x.
    Rows are [cx, cy, half_len, half_wid, z0, z1, albedo, yaw, material]. Version-3 roads keep
    their axis-aligned cars so the version-3 checkpoint still sees exactly its training frames."""
    cy = float(road.centre(x)) + d; yaw = float(road.heading(x)) if TASK_VERSION >= 4 else 0.0
    rows.append([x, cy, 2.1, 0.9, 0.0, 1.5, alb, yaw, mat])
    rows.append([x - (0.1 if forward else -0.1), cy, 0.9, 0.75, 1.5, 2.1, cabin_alb, yaw, mat])


def _boxes_in_world(road, t, cam_x, own=None):
    """All boxes near the camera as (possibly oriented) boxes in world coordinates.
    Road coordinates are turned into world coordinates by adding the centreline offset
    at the box's x, which is exact for the ground and a good approximation for tall boxes
    on a gently curving road."""
    rows = []
    near = road.static[np.abs(road.static[:, 0] - cam_x) < 120]
    for x0, x1, d0, d1, z0, z1, alb in near:
        cx = 0.5*(x0+x1); cy = float(road.centre(cx)) + 0.5*(d0+d1)
        rows.append([cx, cy, 0.5*(x1-x0), 0.5*(d1-d0), z0, z1, alb, 0.0, LAMP if d1-d0 < 0.5 else BUILDING])
    for px, pd in road.parked:
        if -15 < px - cam_x < 120:
            _car_boxes(rows, road, px, pd, 0.9, PARKED, 0.25)
    for ox, od in road.oncoming_positions(t):
        if -15 < ox - cam_x < 120:
            _car_boxes(rows, road, ox, od, 0.15, ONCOMING, 0.3, forward=False)
    for tx, td in road.traffic_positions(t):
        if -15 < tx - cam_x < 120:
            _car_boxes(rows, road, tx, td, 0.6 if TASK_VERSION < 5 else 0.95, TRAFFIC, 0.3 if TASK_VERSION < 5 else 0.25)
    rows = [r + [BOX] for r in rows]
    if own is not None:
        rows += _fly_parts(own, t)
    return np.array(rows, dtype=float) if rows else np.zeros((0, 10))


def _fly_parts(car, t):
    """The driver drawn as a hovering fly the size of the car it stands in for: head with two
    compound eyes, thorax, abdomen, a pair of beating wings and a shadow on the road. Rows are
    [cx, cy, rx, ry, z0, z1, albedo, yaw, material, ELLIPSOID] in world coordinates; the collision
    box of the vehicle is unchanged, this is only what the chase camera sees."""
    c, s = np.cos(car.theta), np.sin(car.theta)
    bob = 0.12*np.sin(7.0*t)
    def at(fx, fy):                       # body-frame offsets (forward, left) to world x, y
        return car.x + c*fx - s*fy, car.y + s*fx + c*fy
    def part(fx, fy, cz, rx, ry, rz, alb, mat, yaw=None):
        x, y = at(fx, fy); cz = cz + bob
        return [x, y, rx, ry, cz-rz, cz+rz, alb, car.theta if yaw is None else yaw, mat, ELLIPSOID]
    flap = 0.35*np.sin(40.0*t)             # wing beat, alternating dihedral
    return [
        part(0.0, 0.0, 0.03, 2.0, 0.9, 0.02, 0.12, FLY_SHADOW),                      # shadow (no bob: it sits on the road)
        part(1.6, 0.0, 1.55, 0.55, 0.6, 0.5, 0.5, FLY_HEAD),                         # head
        part(1.75, 0.45, 1.7, 0.32, 0.32, 0.32, 0.7, FLY_EYE),                        # eyes
        part(1.75, -0.45, 1.7, 0.32, 0.32, 0.32, 0.7, FLY_EYE),
        part(0.5, 0.0, 1.45, 0.85, 0.8, 0.7, 0.6, FLY_BODY),                          # thorax
        part(-1.35, 0.0, 1.15, 1.6, 0.68, 0.5, 0.5, FLY_BODY),                        # abdomen
        part(-0.5, 1.0, 1.95 + flap, 1.55, 0.55, 0.05, 0.9, FLY_WING, car.theta + 0.32),   # wings, swept back
        part(-0.5, -1.0, 1.95 - flap, 1.55, 0.55, 0.05, 0.9, FLY_WING, car.theta - 0.32),
    ]


def _rays(w, h, focal_scale, pitch):
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    f = FOCAL*focal_scale
    d = np.stack([np.ones_like(u, dtype=float), (w/2 - 0.5 - u)/f, (h/2 - 0.5 - v)/f + pitch], axis=-1)
    return d/np.linalg.norm(d, axis=-1, keepdims=True)


_RAY_CACHE = {}


def render_scene(road, t, cam, yaw, pitch, w, h, own=None, detail=False):
    """Shaded grayscale frame from a camera at `cam` = (x, y, z) looking along `yaw` with the
    given pitch. Sky gradient, ground with lane markings and sidewalks, then every box by
    ray-box intersection with a z-buffer; cars are oriented boxes. With detail=True also
    returns the material id, the surface light and the haze factor per pixel for colouring."""
    key = (w, h, round(pitch, 5))
    if key not in _RAY_CACHE:
        _RAY_CACHE[key] = _rays(w, h, w/IMAGE_W, pitch)
    d = _RAY_CACHE[key]
    c, s = np.cos(yaw), np.sin(yaw)
    dx = c*d[..., 0] - s*d[..., 1]; dy = s*d[..., 0] + c*d[..., 1]; dz = d[..., 2]
    ox, oy, oz = cam
    # Sky.
    horizon = np.clip(0.5 + 2.5*dz, 0, 1)
    image = 0.55 + 0.35*horizon
    depth = np.full(d.shape[:2], np.inf)
    material = np.full(d.shape[:2], SKY, dtype=np.uint8)
    light = np.ones(d.shape[:2])
    # Ground.
    ground = dz < -1e-6
    tg = np.where(ground, -oz/np.where(ground, dz, -1.0), 0.0)
    gx, gy = ox + tg*dx, oy + tg*dy
    lat = road.offset(gx, gy)
    along = gx
    g = np.full(d.shape[:2], 0.32); m = np.full(d.shape[:2], GRASS, dtype=np.uint8)   # far ground / grass
    on_road = np.abs(lat) < ROAD_HALF
    g[on_road] = 0.22; m[on_road] = ROAD
    dash = on_road & (np.abs(lat) < 0.12) & ((along % 6.0) < 3.0); g[dash] = 0.85; m[dash] = MARKING
    edge = on_road & (np.abs(np.abs(lat) - (ROAD_HALF-0.15)) < 0.1); g[edge] = 0.8; m[edge] = MARKING
    walk = (np.abs(lat) >= ROAD_HALF) & (np.abs(lat) < 5.6); g[walk] = 0.5; m[walk] = SIDEWALK
    curb = np.abs(np.abs(lat) - ROAD_HALF) < 0.18; g[curb] = 0.65; m[curb] = CURB
    image = np.where(ground, g, image); depth = np.where(ground, tg, depth); material = np.where(ground, m, material)
    # Boxes.
    dirs = np.stack([dx, dy, dz], -1)
    inv0 = 1.0/np.where(np.abs(dirs) < 1e-9, 1e-9, dirs)
    origin0 = np.array([ox, oy, oz])
    light_dir = np.array([0.3, -0.45, 0.84])              # sun: ahead-right and high, for the ellipsoid shading
    for cx, cy, hl, hw, z0, z1, alb, byaw, mat, shape in _boxes_in_world(road, t, ox, own):
        if shape == ELLIPSOID:
            cb, sb = np.cos(byaw), np.sin(byaw)
            rx_, ry_ = ox - cx, oy - cy
            radii = np.array([hl, hw, 0.5*(z1-z0)])
            o_ = np.array([cb*rx_ + sb*ry_, -sb*rx_ + cb*ry_, oz - 0.5*(z0+z1)])/radii
            d_ = np.stack([cb*dx + sb*dy, -sb*dx + cb*dy, dz], -1)/radii
            a = (d_*d_).sum(-1); b = 2*(d_*o_).sum(-1); cc = (o_*o_).sum() - 1.0
            disc = b*b - 4*a*cc
            ok = disc > 0
            tt = np.where(ok, (-b - np.sqrt(np.where(ok, disc, 0.0)))/(2*a), np.inf)
            hit = ok & (tt > 0.05) & (tt < depth)
            if not hit.any():
                continue
            n_ = (o_ + np.where(hit, tt, 0.0)[..., None]*d_)/radii; n_ = n_/np.linalg.norm(n_, axis=-1, keepdims=True).clip(1e-9)
            n_world = np.stack([cb*n_[..., 0] - sb*n_[..., 1], sb*n_[..., 0] + cb*n_[..., 1], n_[..., 2]], -1)
            shade = 0.45 + 0.55*np.clip((n_world*light_dir).sum(-1), 0, 1)
            image = np.where(hit, alb*shade, image); depth = np.where(hit, tt, depth)
            material = np.where(hit, np.uint8(mat), material); light = np.where(hit, shade, light)
            continue
        if byaw != 0.0:
            cb, sb = np.cos(byaw), np.sin(byaw)
            rx, ry = ox - cx, oy - cy
            origin = np.array([cb*rx + sb*ry, -sb*rx + cb*ry, oz])
            ldirs = np.stack([cb*dx + sb*dy, -sb*dx + cb*dy, dz], -1)
            inv = 1.0/np.where(np.abs(ldirs) < 1e-9, 1e-9, ldirs)
            lo, hi = np.array([-hl, -hw, z0]), np.array([hl, hw, z1])
        else:
            origin, inv = origin0, inv0
            lo, hi = np.array([cx-hl, cy-hw, z0]), np.array([cx+hl, cy+hw, z1])
        t1 = (lo - origin)*inv; t2 = (hi - origin)*inv
        tmin = np.maximum(np.minimum(t1, t2).max(-1), 0.05)
        tmax = np.maximum(t1, t2).min(-1)
        hit = (tmax >= tmin) & (tmin < depth)
        if not hit.any():
            continue
        # Which slab was entered last decides the face normal, hence the shading.
        entry = np.argmax(np.minimum(t1, t2), axis=-1)
        shade = np.where(entry == 0, 0.95, np.where(entry == 1, 0.7, 1.0))
        image = np.where(hit, alb*shade, image); depth = np.where(hit, tmin, depth)
        material = np.where(hit, np.uint8(mat), material)
        light = np.where(hit, shade*(alb/0.6 if mat == BUILDING else 1.0), light)
    haze = np.clip(1.0 - depth/160.0, 0.0, 1.0)
    frame = (image*haze + 0.62*(1-haze)).astype(np.float32)
    if detail:
        return {'image': frame, 'material': material, 'light': light.astype(np.float32), 'haze': haze.astype(np.float32)}
    return frame


def render(car, w=IMAGE_W, h=IMAGE_H):
    """What the policy sees: the driver's-eye grayscale frame."""
    return render_scene(car.road, car.t, (car.x, car.y, CAM_HEIGHT), car.theta, PITCH, w, h)


def render_chase(car, w, h, back=10.0, height=4.0, side=1.6, pitch=-0.27):
    """Third-person view for the video: camera behind, above and a little to the left of the
    driver, turned to keep the fly centred, so the fly is seen in three-quarter view."""
    c, s = np.cos(car.theta), np.sin(car.theta)
    cam = (car.x - back*c - side*s, car.y - back*s + side*c, height)
    return render_scene(car.road, car.t, cam, car.theta + np.arctan2(-side, back), pitch, w, h, own=car, detail=True)


def overtaking(car):
    """True while the expert holds the left lane: a slow car is ahead within PULL_OUT m and not yet level."""
    traffic = car.road.traffic_positions(car.t)
    rel = traffic[:, 0] - car.x if len(traffic) else np.zeros(0)
    return bool(np.any((rel > 0.0) & (rel < PULL_OUT)))


def expert(car, lookahead=11.0):
    """Pure pursuit on the right-lane centre. Pulls into the left lane to overtake a slow car
    ahead and returns once level with it; otherwise swerves left inside the lane to clear
    parked cars. Only used for demonstrations."""
    target_offset = LANE_CENTRE
    ax = car.x + lookahead
    if overtaking(car):
        target_offset = LEFT_LANE
    else:
        ahead = car.road.parked[(car.road.parked[:, 0] > car.x - 4) & (car.road.parked[:, 0] < car.x + 30)]
        if len(ahead):
            px, pd = ahead[np.argmin(ahead[:, 0])]
            target_offset = _swerve_offset(pd)
            ax = min(ax, max(px - 2.0, car.x + 5.0))
    ay = car.road.centre(ax) + target_offset/np.cos(car.road.heading(ax))
    ddx, ddy = ax - car.x, ay - car.y
    alpha = (np.arctan2(ddy, ddx) - car.theta + np.pi) % (2*np.pi) - np.pi
    return float(np.clip(np.arctan2(2*WHEELBASE*np.sin(alpha), np.hypot(ddx, ddy)), -MAX_STEER, MAX_STEER))


def rollout(seed, policy, steps=500, obstacles=True):
    road = Road(seed, obstacles=obstacles)
    car = Car(road, np.random.default_rng(seed + 7919))
    lateral, failure = [], None
    for _ in range(steps):
        car.step(policy(render(car), car))
        lateral.append(abs(car.lateral - LANE_CENTRE))
        failure = car.crashed()
        if failure:
            break
    return {'seed': seed, 'steps': len(lateral), 'failure': failure,
            'mean_abs_lateral': float(np.mean(lateral)), 'max_abs_lateral': float(np.max(lateral)),
            'distance_m': float(car.x)}
