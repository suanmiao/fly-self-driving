"""Procedural stylised fly for Blender (bpy). Returns the root Empty; all parts are parented to it.

Local frame: +X forward (head), +Z up. Body length about 1.0 unit before `scale`.
Wings are keyed per frame by `animate_wings(root, n_frames)`; a faint translucent disc suggests the beat.
"""
import math
import bpy
from mathutils import Vector, Euler


def _mat(name, base, rough=0.5, metal=0.0, subsurface=0.0, alpha=1.0, emission=None, bump=None, coat=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes['Principled BSDF']
    bsdf.inputs['Base Color'].default_value = (*base, 1)
    bsdf.inputs['Roughness'].default_value = rough
    bsdf.inputs['Metallic'].default_value = metal
    bsdf.inputs['Alpha'].default_value = alpha
    bsdf.inputs['Coat Weight'].default_value = coat
    if subsurface:
        bsdf.inputs['Subsurface Weight'].default_value = subsurface
        bsdf.inputs['Subsurface Radius'].default_value = (0.05, 0.02, 0.01)
    if emission:
        bsdf.inputs['Emission Color'].default_value = (*emission[0], 1)
        bsdf.inputs['Emission Strength'].default_value = emission[1]
    if bump:  # (texture type, scale, strength)
        kind, scale, strength = bump
        tex = nt.nodes.new('ShaderNodeTexVoronoi' if kind == 'voronoi' else 'ShaderNodeTexNoise')
        tex.inputs['Scale'].default_value = scale
        b = nt.nodes.new('ShaderNodeBump')
        b.inputs['Strength'].default_value = strength
        nt.links.new(tex.outputs['Distance' if kind == 'voronoi' else 'Fac'], b.inputs['Height'])
        nt.links.new(b.outputs['Normal'], bsdf.inputs['Normal'])
    if alpha < 1:
        m.blend_method = 'BLEND'
        try:
            m.surface_render_method = 'BLENDED'
        except AttributeError:
            pass
    return m


def _sphere(name, loc, scale, mat, parent, rot=(0, 0, 0), segs=32):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segs, ring_count=segs // 2, radius=1.0, location=(0, 0, 0))
    o = bpy.context.active_object
    o.name = name
    o.scale = scale
    o.location = loc
    o.rotation_euler = Euler(rot)
    o.data.materials.append(mat)
    o.parent = parent
    bpy.ops.object.shade_smooth()
    return o


def _cyl(name, p0, p1, radius, mat, parent):
    p0, p1 = Vector(p0), Vector(p1)
    d = p1 - p0
    bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=radius, depth=d.length, location=(0, 0, 0))
    o = bpy.context.active_object
    o.name = name
    o.location = (p0 + p1) / 2
    o.rotation_euler = d.to_track_quat('Z', 'Y').to_euler()
    o.data.materials.append(mat)
    o.parent = parent
    bpy.ops.object.shade_smooth()
    return o


def _wing(name, side, mat, parent):
    """Flat wing with a rounded outline, hinged at the thorax."""
    import bmesh
    pts = [(0, 0), (0.18, 0.06), (0.42, 0.10), (0.68, 0.10), (0.90, 0.06), (1.0, 0.0),
           (0.92, -0.06), (0.70, -0.11), (0.40, -0.11), (0.15, -0.06)]
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    verts = [bm.verts.new((x * 0.62, side * (0.04 + 0.08 + y * 0) + y * side * 0 + y * 1.0, 0)) for x, y in pts]
    # rotate outline so the wing sweeps sideways (along ±Y) and slightly back (−X)
    for v in verts:
        x, y, z = v.co
        v.co = Vector((-0.55 * x + 0.15 * y * side, side * (x * 0.75) + y * 0.5 * side * -1 + 0.0, 0))
    bm.faces.new(verts)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.to_mesh(mesh)
    bm.free()
    o = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(o)
    o.data.materials.append(mat)
    o.parent = parent
    return o


def build_fly(name='fly', scale=1.0):
    root = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(root)

    body_col = (0.16, 0.13, 0.09)
    m_thorax = _mat('fly_thorax', (0.13, 0.085, 0.045), rough=0.5, subsurface=0.1, bump=('noise', 40, 0.3), coat=0.4)
    m_abd = _mat('fly_abdomen', (0.55, 0.30, 0.09), rough=0.4, subsurface=0.2, coat=0.5)
    m_stripe = _mat('fly_stripe', (0.06, 0.04, 0.025), rough=0.5, coat=0.4)
    m_eye = _mat('fly_eye', (0.85, 0.05, 0.03), rough=0.2, coat=1.0, bump=('voronoi', 60, 0.4))
    m_leg = _mat('fly_leg', (0.07, 0.05, 0.03), rough=0.6)
    m_wing = _mat('fly_wing', (0.85, 0.9, 1.0), rough=0.05, alpha=0.32, coat=1.0)
    m_blur = _mat('fly_wingblur', (0.9, 0.93, 1.0), rough=0.3, alpha=0.035)
    m_halt = _mat('fly_haltere', (0.7, 0.6, 0.3), rough=0.5)

    # thorax (centre), head (front), abdomen (rear)
    thorax = _sphere('thorax', (0.05, 0, 0), (0.30, 0.25, 0.24), m_thorax, root)
    head = _sphere('head', (0.40, 0, 0.02), (0.15, 0.17, 0.16), m_thorax, root)
    abd = _sphere('abdomen', (-0.42, 0, -0.03), (0.40, 0.22, 0.19), m_abd, root)
    for i, x in enumerate((-0.30, -0.44, -0.58, -0.70)):  # abdominal bands
        r = 0.22 * math.sqrt(max(0.0, 1 - ((x + 0.42) / 0.40) ** 2))
        _sphere(f'band{i}', (x, 0, -0.03), (0.035, r * 1.005, r * 0.87), m_stripe, root, segs=24)
    # compound eyes
    for s in (1, -1):
        _sphere(f'eye{"L" if s > 0 else "R"}', (0.44, s * 0.115, 0.06), (0.10, 0.085, 0.11), m_eye, root, rot=(0, 0, s * 0.35))
    _sphere('proboscis', (0.50, 0, -0.10), (0.03, 0.03, 0.06), m_leg, root, segs=12)
    for s in (1, -1):  # antennae
        _cyl(f'ant{s}', (0.52, s * 0.03, 0.08), (0.60, s * 0.06, 0.13), 0.008, m_leg, root)
    # halteres
    for s in (1, -1):
        _cyl(f'halt_stalk{s}', (-0.12, s * 0.20, 0.05), (-0.20, s * 0.30, 0.10), 0.008, m_halt, root)
        _sphere(f'halt_knob{s}', (-0.20, s * 0.30, 0.10), (0.025, 0.025, 0.025), m_halt, root, segs=12)
    # six legs: coxa on thorax, femur out+up, tibia down, tarsus forward
    leg_roots = [(0.22, 0.17, -0.10), (0.05, 0.22, -0.12), (-0.12, 0.20, -0.12)]
    for s in (1, -1):
        for i, (x, y, z) in enumerate(leg_roots):
            y *= s
            knee = (x + (0.12 if i == 0 else (0.0 if i == 1 else -0.14)), y + s * 0.22, z + 0.12)
            ankle = (knee[0] + (0.08 if i == 0 else (0.02 if i == 1 else -0.10)), y + s * 0.34, z - 0.22)
            toe = (ankle[0] + 0.10, y + s * 0.36, z - 0.30)
            _cyl(f'femur{s}{i}', (x, y, z), knee, 0.016, m_leg, root)
            _cyl(f'tibia{s}{i}', knee, ankle, 0.012, m_leg, root)
            _cyl(f'tarsus{s}{i}', ankle, toe, 0.009, m_leg, root)
    # wings hinged at thorax top
    wings = []
    for s in (1, -1):
        hinge = bpy.data.objects.new(f'wing_hinge{"L" if s > 0 else "R"}', None)
        bpy.context.collection.objects.link(hinge)
        hinge.parent = root
        hinge.location = (0.02, s * 0.14, 0.20)
        w = _wing(f'wing{"L" if s > 0 else "R"}', s, m_wing, hinge)
        w.scale = (1.15, 1.15, 1.0)
        wings.append(hinge)
        blur = _sphere(f'wingblur{s}', (-0.22, s * 0.40, 0.26), (0.30, 0.24, 0.015), m_blur, root, segs=24)
    root.scale = (scale, scale, scale)
    root['wing_hinges'] = [h.name for h in wings]
    return root


def animate_wings(root, n_frames, amplitude=0.55, rest=0.35):
    """Alternate the wing stroke every frame (beat is far above the frame rate, so any phase is honest)."""
    for name in root['wing_hinges']:
        h = bpy.data.objects[name]
        side = 1 if name.endswith('L') else -1
        for f in range(1, n_frames + 1):
            phase = 1 if f % 2 else -1
            h.rotation_euler = Euler((side * (rest + phase * amplitude) * 0.6, 0.15 * phase, side * 0.25))
            h.keyframe_insert('rotation_euler', frame=f)
