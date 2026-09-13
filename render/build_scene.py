"""Blender replay renderer for the fly street-driving task.

Usage (headless):
  blender -b -noaudio --python build_scene.py -- --log episode.json --out frames/ [--frames 0:300] [--res 1600x900] [--quick]

Input: an episode log (see synth_episode.py for the schema). Everything visual is built here from
Kenney CC0 kits (assets/) plus the procedural fly in fly_model.py. The physics, the brain and the
64x32 view the brain sees are untouched: this only replays recorded states.

Log schema (metres, radians, +X along the street, +Y left):
{
  "dt": 0.05, "lane_half_width": 3.5,
  "road": {"x": [...], "y": [...]},                       # centreline samples, 1 m apart, covering the drive
  "parked": [{"x":..,"y":..,"heading":..,"kind":"sedan"}],
  "traffic": [{"kind":"suv","poses":[[x,y,heading],...]}], # one pose per tick, same length as fly.poses
  "fly": {"poses": [[x,y,heading,steer],...]},
  "collision": {"tick": int or null, "with": "oncoming"|"slow"|"parked"|"off_road"|null}
}
"""
import argparse, json, math, os, random, sys
import numpy as np
import bpy
from mathutils import Vector, Euler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fly_model  # noqa: E402

ASSETS = os.path.join(HERE, 'assets')
CARS = os.path.join(ASSETS, 'car-kit', 'Models', 'GLB format')
BUILD_C = os.path.join(ASSETS, 'city-kit-commercial', 'Models', 'GLB format')
BUILD_S = os.path.join(ASSETS, 'city-kit-suburban', 'Models', 'GLB format')
ROADS = os.path.join(ASSETS, 'city-kit-roads', 'Models', 'GLB format')
NATURE = os.path.join(ASSETS, 'nature-kit', 'Models', 'GLTF format')
PEOPLE = os.path.join(ASSETS, 'mini-characters', 'Models', 'GLB format')


def build_dog_collection():
    """Low-poly dog (0.9 long, 0.4 wide, 0.6 tall, base at z=0) in a collection for instancing; +X forward."""
    col = bpy.data.collections.new('lib_dog')
    bpy.context.scene.collection.children.link(col)
    m_fur = bpy.data.materials.new('dog_fur'); m_fur.use_nodes = True
    m_fur.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.55, 0.36, 0.18, 1)
    m_dark = bpy.data.materials.new('dog_dark'); m_dark.use_nodes = True
    m_dark.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.08, 0.06, 0.05, 1)

    def part(name, loc, scale, mat=m_fur, prim='cube'):
        if prim == 'cube':
            bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
        else:
            bpy.ops.mesh.primitive_uv_sphere_add(segments=12, ring_count=8, radius=0.5, location=loc)
        o = bpy.context.active_object
        o.name = name; o.scale = scale; o.data.materials.append(mat)
        for c in list(o.users_collection):
            c.objects.unlink(o)
        col.objects.link(o)
        return o
    part('dog_body', (0.0, 0, 0.36), (0.55, 0.28, 0.26))
    part('dog_chest', (0.22, 0, 0.38), (0.22, 0.30, 0.30), prim='sphere')
    part('dog_head', (0.40, 0, 0.50), (0.22, 0.22, 0.20), prim='sphere')
    part('dog_snout', (0.52, 0, 0.45), (0.14, 0.12, 0.10))
    part('dog_nose', (0.59, 0, 0.47), (0.04, 0.05, 0.04), m_dark)
    for s in (1, -1):
        part(f'dog_ear{s}', (0.36, s * 0.10, 0.60), (0.08, 0.04, 0.14), m_dark)
        part(f'dog_eye{s}', (0.48, s * 0.07, 0.53), (0.03, 0.03, 0.03), m_dark, prim='sphere')
        part(f'dog_legF{s}', (0.20, s * 0.10, 0.12), (0.09, 0.09, 0.24))
        part(f'dog_legB{s}', (-0.20, s * 0.10, 0.12), (0.09, 0.09, 0.24))
    part('dog_tail', (-0.32, 0, 0.50), (0.06, 0.05, 0.22))
    vl = bpy.context.view_layer.layer_collection.children.get(col.name)
    if vl is not None:
        vl.exclude = True
    col.instance_offset = (0, 0, 0)
    return col

CAR_SCALE = 1.75          # Kenney sedan is 2.55 long -> ~4.5 m
BUILDING_SCALE = 9.0      # Kenney buildings are ~1 unit -> ~9-12 m
TREE_SCALE = 5.0
KIND_FILES = {'sedan': 'sedan.glb', 'suv': 'suv.glb', 'taxi': 'taxi.glb', 'van': 'van.glb',
              'hatchback': 'hatchback-sports.glb', 'police': 'police.glb', 'sports': 'sedan-sports.glb',
              'race': 'race.glb', 'delivery': 'delivery.glb', 'luxury': 'suv-luxury.glb'}


# ---------------------------------------------------------------- helpers
def parse_args():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument('--log', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--frames', default=None, help='start:end (ticks)')
    p.add_argument('--res', default='1600x900')
    p.add_argument('--quick', action='store_true', help='low samples for previews')
    p.add_argument('--every', type=int, default=1, help='render every Nth tick')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--build-only', action='store_true', help='build and save the .blend, no render')
    return p.parse_args(argv)


def import_glb(path, name):
    """Import a glb into its own collection; return the collection (for instancing)."""
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    new = [o for o in bpy.data.objects if o not in before]
    # Kenney character glbs ship an unparented helper Icosphere; drop it
    for o in [o for o in new if o.type == 'MESH' and o.parent is None and o.name.startswith('Icosphere')]:
        bpy.data.objects.remove(o, do_unlink=True)
        new.remove(o)
    col = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(col)
    for o in new:
        for c in list(o.users_collection):
            c.objects.unlink(o)
        col.objects.link(o)
    # centre the footprint on the origin and put the base at z=0 via the collection's instance offset
    # evaluated bounds: for rigged characters this is the posed mesh, not the bind pose
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    pts = []
    for o in new:
        if o.type != 'MESH':
            continue
        ev = o.evaluated_get(dg)
        mw = ev.matrix_world
        pts += [mw @ v.co for v in ev.data.vertices]   # posed vertices, so rigged characters measure right
    if pts:
        lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
        hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
        col.instance_offset = ((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z)
        col['size'] = list(hi - lo)
    return col


def asset_size(path, name):
    key = os.path.abspath(path)
    if key not in _LIB:
        instance(path, name, (0, 0, -1000), 0, 1.0, bpy.context.scene.collection)
    return _LIB[key]['size']


_LIB = {}


def instance(path, name, loc, rot_z, scale, parent_col):
    key = os.path.abspath(path)
    if key not in _LIB:
        col = import_glb(path, f'lib_{name}')
        # exclude the source collection from the view layer: instances still render, the source does not
        vl = bpy.context.view_layer.layer_collection.children.get(col.name)
        if vl is not None:
            vl.exclude = True
        _LIB[key] = col
    inst = bpy.data.objects.new(f'{name}_inst', None)
    inst.instance_type = 'COLLECTION'
    inst.instance_collection = _LIB[key]
    inst.location = loc
    inst.rotation_euler = Euler((0, 0, rot_z))
    inst.scale = (scale, scale, scale)
    parent_col.objects.link(inst)
    return inst


def material(name, base, rough=0.6, metal=0.0, tex=None, emission=None):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes['Principled BSDF']
    bsdf.inputs['Base Color'].default_value = (*base, 1)
    bsdf.inputs['Roughness'].default_value = rough
    bsdf.inputs['Metallic'].default_value = metal
    if emission:
        bsdf.inputs['Emission Color'].default_value = (*emission[0], 1)
        bsdf.inputs['Emission Strength'].default_value = emission[1]
    if tex is not None:
        img = nt.nodes.new('ShaderNodeTexImage')
        img.image = tex
        img.extension = 'REPEAT'
        img.interpolation = 'Closest'
        nt.links.new(img.outputs['Color'], bsdf.inputs['Base Color'])
    return m


def road_texture(period_m=6.0, px_per_m=32):
    """Tileable asphalt with a dashed centre line and solid edge lines. u = x/period_m, v across."""
    w, h = 256, int(period_m * px_per_m)
    img = np.zeros((h, w, 4), dtype=np.float32)
    rng = np.random.default_rng(1)
    asphalt = 0.16 + 0.03 * rng.random((h, w, 1))
    img[..., :3] = asphalt
    img[..., 3] = 1
    v = np.linspace(0, 1, w)
    edge = (np.abs(v - 0.5) > 0.475) & (np.abs(v - 0.5) < 0.495)
    img[:, edge, :3] = 0.85
    centre = np.abs(v - 0.5) < 0.012
    dash = (np.arange(h) % h) < h * 0.5
    img[np.ix_(dash, centre)] = (0.9, 0.85, 0.5, 1)
    tex = bpy.data.images.new('road_tex', w, h, alpha=True, float_buffer=True)
    tex.pixels = img.ravel().tolist()
    return tex


def ribbon(name, xs, ys, headings, half_left, half_right, z, mat, u_scale=1.0, col=None):
    """Mesh strip along a polyline: offsets are signed distances to the left (+) and right (-)."""
    verts, faces, uvs = [], [], []
    for i, (x, y, h) in enumerate(zip(xs, ys, headings)):
        nx, ny = -math.sin(h), math.cos(h)
        verts.append((x + nx * half_right, y + ny * half_right, z))
        verts.append((x + nx * half_left, y + ny * half_left, z))
        if i:
            faces.append((2 * i - 2, 2 * i - 1, 2 * i + 1, 2 * i))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    uv = mesh.uv_layers.new(name='UVMap')
    for poly in mesh.polygons:
        for li in poly.loop_indices:
            vi = mesh.loops[li].vertex_index
            row, side = vi // 2, vi % 2
            uv.data[li].uv = (side, row * u_scale)
    o = bpy.data.objects.new(name, mesh)
    (col or bpy.context.scene.collection).objects.link(o)
    o.data.materials.append(mat)
    return o


# ---------------------------------------------------------------- scene
def build(log, args):
    rng = random.Random(args.seed)
    scene = bpy.context.scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.preferences.edit.keyframe_new_interpolation_type = 'LINEAR'   # keys are per tick; no easing
    scene = bpy.context.scene
    world_col = bpy.data.collections.new('street')
    scene.collection.children.link(world_col)

    xs, ys = np.asarray(log['road']['x'], float), np.asarray(log['road']['y'], float)
    heads = np.arctan2(np.gradient(ys), np.gradient(xs))
    half = float(log['lane_half_width'])

    # ground, road, curbs, sidewalks
    m_grass = material('grass', (0.30, 0.42, 0.18), rough=0.9)
    bpy.ops.mesh.primitive_plane_add(size=1, location=((xs[0] + xs[-1]) / 2, 0, -0.02))
    ground = bpy.context.active_object
    ground.scale = ((xs[-1] - xs[0]) + 400, 400, 1)
    ground.data.materials.append(m_grass)
    m_road = material('road', (0.2, 0.2, 0.2), rough=0.85, tex=road_texture())
    m_curb = material('curb', (0.72, 0.70, 0.66), rough=0.7)
    m_walk = material('sidewalk', (0.62, 0.60, 0.57), rough=0.8)
    ribbon('road', xs, ys, heads, half + 0.1, -(half + 0.1), 0.0, m_road, u_scale=1 / 6.0, col=world_col)
    for s in (1, -1):
        ribbon(f'curb{s}', xs, ys, heads, s * (half + 0.45), s * (half + 0.1), 0.13, m_curb, col=world_col)
        ribbon(f'walk{s}', xs, ys, heads, s * (half + 3.0), s * (half + 0.45), 0.13, m_walk, col=world_col)

    # buildings, trees, lamps along both sides
    buildings = [os.path.join(BUILD_C, f) for f in sorted(os.listdir(BUILD_C)) if f.startswith('building-') and 'skyscraper' not in f] \
        + [os.path.join(BUILD_S, f) for f in sorted(os.listdir(BUILD_S)) if f.startswith('building-')]
    trees = [os.path.join(NATURE, f) for f in sorted(os.listdir(NATURE)) if f.startswith('tree_') and f.endswith('.glb') and 'fall' not in f]
    lamp = os.path.join(ROADS, 'light-curved.glb')
    def road_frame(x_along):
        k = int(np.clip(np.searchsorted(xs, x_along), 0, len(xs) - 1))
        return k, heads[k], -math.sin(heads[k]), math.cos(heads[k])

    has_lamps = False
    if log.get('buildings'):
        # place a Kenney building on every recorded box footprint (same layout the brain saw)
        for bx in log['buildings']:
            xm = 0.5 * (bx['x0'] + bx['x1'])
            lat = 0.5 * (bx['lateral0'] + bx['lateral1'])
            k, h, nx, ny = road_frame(xm)
            pos = (xs[k] + nx * lat, ys[k] + ny * lat, 0)
            if bx.get('kind') == 'lamp_post':
                has_lamps = True
                sz = asset_size(lamp, 'lamp')
                # lamp arm reaches over the road: kit lamp arm points +Y after import; face the road
                instance(lamp, 'lamp', pos, h + (math.pi if lat > 0 else 0), (bx['z1'] - bx['z0']) / sz[2], world_col)
                continue
            if bx.get('kind') != 'building':
                continue
            b = rng.choice(buildings)
            sx, sy, sz = asset_size(b, os.path.basename(b)[:-4])
            wdt, depth = bx['x1'] - bx['x0'], abs(bx['lateral1'] - bx['lateral0'])
            sc = float(np.clip(min(wdt / sx, depth / sy), 4.0, 20.0))
            # keep the near face at the recorded near edge so nothing intrudes on the road
            near = min(abs(bx['lateral0']), abs(bx['lateral1']))
            lat_c = math.copysign(near + sy * sc / 2, lat)
            pos = (xs[k] + nx * lat_c, ys[k] + ny * lat_c, 0)
            facing = h + (math.pi / 2 if lat < 0 else -math.pi / 2)
            instance(b, os.path.basename(b)[:-4], pos, facing + math.pi / 2, sc, world_col)
    step = 14.0
    x = xs[0]
    i = 0
    while x < xs[-1]:
        k, h, nx, ny = road_frame(x)
        for s in (1, -1):
            off = half + 3.6 + rng.uniform(4, 9)
            if not log.get('buildings') and rng.random() < 0.8:
                b = rng.choice(buildings)
                sc = BUILDING_SCALE * rng.uniform(0.9, 1.4)
                instance(b, os.path.basename(b)[:-4], (xs[k] + nx * s * off, ys[k] + ny * s * off, 0), h + (math.pi if s > 0 else 0) + math.pi / 2, sc, world_col)
            if rng.random() < 0.6:
                t = rng.choice(trees)
                toff = half + 2.0
                instance(t, os.path.basename(t)[:-4], (xs[k] + nx * s * toff + math.cos(h) * rng.uniform(-4, 4), ys[k] + ny * s * toff + math.sin(h) * rng.uniform(-4, 4), 0.13), rng.uniform(0, 6.3), TREE_SCALE * rng.uniform(0.8, 1.3), world_col)
        if i % 2 == 0 and not has_lamps:
            s = 1 if i % 4 == 0 else -1
            loff = half + 0.9
            instance(lamp, 'lamp', (xs[k] + nx * s * loff, ys[k] + ny * s * loff, 0.13), h + (0 if s > 0 else math.pi), 7.0, world_col)
        x += step
        i += 1

    # parked cars
    for j, p in enumerate(log.get('parked', [])):
        f = KIND_FILES.get(p.get('kind', 'sedan'), 'sedan.glb')
        instance(os.path.join(CARS, f), f'parked{j}', (p['x'], p['y'], 0), p['heading'] - math.pi / 2, CAR_SCALE, world_col)

    # moving traffic
    movers = []
    for j, t in enumerate(log.get('traffic', [])):
        f = KIND_FILES.get(t.get('kind', 'suv'), 'suv.glb')
        inst = instance(os.path.join(CARS, f), f'traffic{j}', (0, 0, 0), 0, CAR_SCALE, world_col)
        movers.append((inst, t['poses']))

    # crossers (v6): pedestrians from Kenney mini-characters, dogs procedural
    crossers = log.get('crossers') or []
    n_cross = max((len(r) for r in crossers), default=0)
    kinds_meta = log.get('crosser_kinds', {})
    if isinstance(kinds_meta, list):   # v6 log: [{'name', 'size_m', 'speed_mps'}, ...]
        kinds_meta = {k['name']: {'size': k.get('size_m', k.get('size'))} for k in kinds_meta}
    cross_objs = []
    if n_cross:
        people = [os.path.join(PEOPLE, f) for f in sorted(os.listdir(PEOPLE)) if f.startswith('character-') and f.endswith('.glb')]
        dog_col = build_dog_collection()
        for j in range(n_cross):
            kind = next((r[j][2] for r in crossers if len(r) > j), 'pedestrian')
            size = kinds_meta.get(kind, {}).get('size', [0.5, 0.5, 1.75] if kind == 'pedestrian' else [0.9, 0.4, 0.6])
            if kind == 'pedestrian':
                p = rng.choice(people)
                sz = asset_size(p, os.path.basename(p)[:-4])
                # Kenney characters are chibi (head is half the height); 0.85x reads as adult-sized next to the kit cars
                inst = instance(p, f'crosser{j}', (0, 0, -100), 0, 0.85 * size[2] / sz[2], world_col)
            else:
                inst = bpy.data.objects.new(f'crosser{j}_inst', None)
                inst.instance_type = 'COLLECTION'
                inst.instance_collection = dog_col
                inst.scale = (size[0] / 0.9, size[1] / 0.4, size[2] / 0.6)
                world_col.objects.link(inst)
            cross_objs.append((inst, kind))

    # the fly
    fly = fly_model.build_fly('fly', scale=2.2)
    fly.location = (0, 0, 0)

    # camera + lights
    cam_data = bpy.data.cameras.new('cam')
    cam_data.lens = 32
    cam_data.dof.use_dof = True
    cam_data.dof.focus_object = fly
    cam_data.dof.aperture_fstop = 5.6
    cam = bpy.data.objects.new('cam', cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    sun_data = bpy.data.lights.new('sun', 'SUN')
    sun_data.energy = 6.0
    sun_data.color = (1.0, 0.93, 0.82)
    sun_data.angle = math.radians(1.5)
    sun = bpy.data.objects.new('sun', sun_data)
    sun.rotation_euler = Euler((math.radians(48), math.radians(12), math.radians(-60)))
    scene.collection.objects.link(sun)
    world = bpy.data.worlds.new('world')
    scene.world = world
    world.use_nodes = True
    wn = world.node_tree
    sky = wn.nodes.new('ShaderNodeTexSky')
    sky.sky_type = "MULTIPLE_SCATTERING"
    sky.sun_elevation = math.radians(35)
    sky.sun_rotation = math.radians(145)
    sky.sun_intensity = 0.3
    sky.altitude = 200
    bg = wn.nodes['Background']
    bg.inputs['Strength'].default_value = 0.35
    wn.links.new(sky.outputs['Color'], bg.inputs['Color'])

    # animation: fly + traffic + chase camera
    poses = log['fly']['poses']
    n = len(poses)
    t0, t1 = (0, n)
    if args.frames:
        a, b = args.frames.split(':')
        t0, t1 = int(a or 0), int(b or n)
    ticks = list(range(t0, min(t1, n), args.every))
    scene.frame_start, scene.frame_end = 1, len(ticks)
    fly_model.animate_wings(fly, len(ticks))
    cam_pos = None
    for fi, tk in enumerate(ticks, start=1):
        x, y, h, steer = poses[tk][:4]
        hover = 0.55 + 0.03 * math.sin(tk * 0.9)
        fly.location = (x, y, hover)
        fly.rotation_euler = Euler((-steer * 0.6, 0.06, h))   # bank into the turn
        fly.keyframe_insert('location', frame=fi)
        fly.keyframe_insert('rotation_euler', frame=fi)
        for inst, tp in movers:
            px, py, ph = tp[min(tk, len(tp) - 1)][:3]
            inst.location = (px, py, 0)
            inst.rotation_euler = Euler((0, 0, ph - math.pi / 2))
            inst.keyframe_insert('location', frame=fi)
            inst.keyframe_insert('rotation_euler', frame=fi)
        for j, (inst, kind) in enumerate(cross_objs):
            row = crossers[min(tk, len(crossers) - 1)]
            if j < len(row):
                cx, cy = row[j][0], row[j][1]
                prev = crossers[max(tk - 1, 0)]
                if j < len(prev) and (abs(prev[j][0] - cx) + abs(prev[j][1] - cy)) > 1e-3:
                    face = math.atan2(cy - prev[j][1], cx - prev[j][0])
                else:
                    face = h + (math.pi / 2 if row[j][3] < 0 else -math.pi / 2)   # waiting at the kerb, facing the road
                bob = 0.03 * math.sin(tk * 1.2) if kind == 'pedestrian' else 0.0
                inst.location = (cx, cy, 0.13 + bob)
                inst.rotation_euler = Euler((0, 0, face - math.pi / 2 if kind == 'pedestrian' else face))
            else:
                inst.location = (0, 0, -100)
            inst.keyframe_insert('location', frame=fi)
            inst.keyframe_insert('rotation_euler', frame=fi)
        # chase camera: behind and above, smoothed
        back = 7.5
        target = Vector((x - back * math.cos(h) + 1.2 * math.sin(h), y - back * math.sin(h) - 1.2 * math.cos(h), 3.2))
        cam_pos = target if cam_pos is None else cam_pos.lerp(target, 0.25)
        cam.location = cam_pos
        look = Vector((x + 4 * math.cos(h), y + 4 * math.sin(h), 0.6))
        cam.rotation_euler = (look - cam_pos).to_track_quat('-Z', 'Y').to_euler()
        cam.keyframe_insert('location', frame=fi)
        cam.keyframe_insert('rotation_euler', frame=fi)

    # render settings
    w, hgt = (int(v) for v in args.res.split('x'))
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x, scene.render.resolution_y = w, hgt
    scene.render.fps = 20
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = os.path.join(args.out, 'frame_')
    ee = scene.eevee
    ee.taa_render_samples = 8 if args.quick else 32
    for attr, val in (('use_shadows', True), ('use_raytracing', not args.quick), ('shadow_ray_count', 2), ('shadow_step_count', 4)):
        if hasattr(ee, attr):
            setattr(ee, attr, val)
    scene.view_settings.view_transform = 'AgX'
    scene.view_settings.look = 'AgX - Punchy'
    scene.view_settings.exposure = -1.0
    scene.render.film_transparent = False
    os.makedirs(args.out, exist_ok=True)
    return scene


def main():
    args = parse_args()
    log = json.load(open(args.log))
    scene = build(log, args)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(args.out, 'scene.blend'))
    print(f'SCENE_BUILT objects={len(bpy.data.objects)} frames={scene.frame_start}-{scene.frame_end}', flush=True)
    if not args.build_only:
        bpy.ops.render.render(animation=True)


if __name__ == '__main__':
    main()
