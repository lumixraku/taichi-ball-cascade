import taichi as ti
import math

# precompute sin/cos for baffle angles
_S, _C = math.sin(0.5), math.cos(0.5)   # ~28.6° — steep enough to slide
_SB, _CB = math.sin(0.45), math.cos(0.45)  # back row angle

ti.init(arch=ti.gpu, random_seed=42)

NUM_BALLS = 1200
SPAWN_RATE = 15
BALL_RADIUS = 0.038
GRAVITY = ti.Vector([0.0, -7.0, 0.0])
DT = 0.0012
SUBSTEPS = 6
RESTITUTION = 0.2
FRICTION = 0.18           # low enough that balls slide on 28° slope

pos = ti.Vector.field(3, float, NUM_BALLS)
vel = ti.Vector.field(3, float, NUM_BALLS)
ball_rgb = ti.Vector.field(3, float, NUM_BALLS)
alive = ti.field(int, NUM_BALLS)
active_count = ti.field(int, ())

NUM_OBS = 20
obs_center = ti.Vector.field(3, float, NUM_OBS)
obs_normal = ti.Vector.field(3, float, NUM_OBS)
obs_w = ti.field(float, NUM_OBS)
obs_h = ti.field(float, NUM_OBS)

obs_vtx = ti.Vector.field(3, float, NUM_OBS * 4)
obs_idx = ti.field(int, NUM_OBS * 8)

palette = ti.Vector.field(3, float, 6)


@ti.kernel
def init():
    palette[0] = ti.Vector([0.88, 0.20, 0.12])
    palette[1] = ti.Vector([0.95, 0.55, 0.10])
    palette[2] = ti.Vector([0.18, 0.58, 0.92])
    palette[3] = ti.Vector([0.92, 0.75, 0.12])
    palette[4] = ti.Vector([0.22, 0.72, 0.42])
    palette[5] = ti.Vector([0.72, 0.18, 0.55])

    for i in range(NUM_BALLS):
        pos[i] = ti.Vector([0.0, -999.0, 0.0])
        vel[i] = ti.Vector([0.0, 0.0, 0.0])
        alive[i] = 0
        ci = i % 6
        base = palette[ci]
        ball_rgb[i] = base + ti.Vector([
            ti.random() * 0.08 - 0.04,
            ti.random() * 0.08 - 0.04,
            ti.random() * 0.08 - 0.04,
        ])
    active_count[None] = 0

    # ── main cascade: 7 staggered baffles, each ~half the container width ──
    # Each creates a clear gap so balls roll → drop → hit next baffle
    # y, sign(angle), x_center, half_width, half_height
    rows = [
        (0.72,  1, -0.45, 0.65, 0.20),   # sloping down to the RIGHT
        (0.48, -1,  0.45, 0.65, 0.20),   # sloping down to the LEFT
        (0.24,  1, -0.42, 0.68, 0.20),
        (0.00, -1,  0.42, 0.68, 0.20),
        (-0.24,  1, -0.45, 0.65, 0.20),
        (-0.48, -1,  0.45, 0.65, 0.20),
        (-0.72,  1, -0.42, 0.68, 0.20),
    ]

    for k in ti.static(range(len(rows))):
        y, sign, sx, w, h = rows[k]
        obs_center[k] = ti.Vector([sx, y, 0.0])
        obs_normal[k] = ti.Vector([_S * sign, _C, 0.0]).normalized()
        obs_w[k] = w
        obs_h[k] = h

    # back row (z = -0.55)
    b_rows = [
        (0.60, -1, 0.40, 0.42, 0.15),
        (0.36,  1, -0.40, 0.42, 0.15),
        (0.12, -1, 0.38, 0.44, 0.15),
        (-0.12,  1, -0.38, 0.44, 0.15),
        (-0.36, -1, 0.40, 0.42, 0.15),
        (-0.60,  1, -0.40, 0.42, 0.15),
    ]
    for k in ti.static(range(len(b_rows))):
        y, sign, sx, w, h = b_rows[k]
        idx = 7 + k
        obs_center[idx] = ti.Vector([sx, y, -0.55])
        obs_normal[idx] = ti.Vector([_SB * sign, _CB, 0.0]).normalized()
        obs_w[idx] = w
        obs_h[idx] = h

    # floor
    obs_center[13] = ti.Vector([0.0, -1.15, 0.0])
    obs_normal[13] = ti.Vector([0.0, 1.0, 0.0])
    obs_w[13] = 1.15; obs_h[13] = 0.0

    # walls
    obs_center[14] = ti.Vector([-1.15, 0.0, 0.0])
    obs_normal[14] = ti.Vector([1.0, 0.0, 0.0])
    obs_w[14] = 1.0; obs_h[14] = 0.0

    obs_center[15] = ti.Vector([1.15, 0.0, 0.0])
    obs_normal[15] = ti.Vector([-1.0, 0.0, 0.0])
    obs_w[15] = 1.0; obs_h[15] = 0.0

    obs_center[16] = ti.Vector([0.0, 0.0, -1.15])
    obs_normal[16] = ti.Vector([0.0, 0.0, 1.0])
    obs_w[16] = 1.0; obs_h[16] = 0.0

    obs_center[17] = ti.Vector([0.0, 0.0, 1.15])
    obs_normal[17] = ti.Vector([0.0, 0.0, -1.0])
    obs_w[17] = 1.0; obs_h[17] = 0.0

    # narrow back walls at z = -0.55 to create a corridor
    obs_center[18] = ti.Vector([0.0, 0.0, -1.0])
    obs_normal[18] = ti.Vector([0.0, 0.0, 1.0])
    obs_w[18] = 1.0; obs_h[18] = 0.0

    obs_center[19] = ti.Vector([0.0, 0.0, -0.1])
    obs_normal[19] = ti.Vector([0.0, 0.0, -1.0])
    obs_w[19] = 1.0; obs_h[19] = 0.0

    # compute quad vertices for each obstacle
    for i in range(NUM_OBS):
        n = obs_normal[i]
        ref = ti.Vector([0.0, 1.0, 0.0])
        if abs(n.dot(ref)) > 0.999:
            ref = ti.Vector([1.0, 0.0, 0.0])
        u = n.cross(ref).normalized() * obs_w[i]
        v = n.cross(u).normalized() * obs_h[i]
        c = obs_center[i]
        obs_vtx[i * 4 + 0] = c - u - v
        obs_vtx[i * 4 + 1] = c + u - v
        obs_vtx[i * 4 + 2] = c + u + v
        obs_vtx[i * 4 + 3] = c - u + v
        b = i * 8
        for k in ti.static(range(4)):
            obs_idx[b + k * 2] = i * 4 + k
            obs_idx[b + k * 2 + 1] = i * 4 + (k + 1) % 4


@ti.func
def plane_collide(i: int, c: ti.template(), n: ti.template()):
    d = (pos[i] - c).dot(n)
    if d < BALL_RADIUS:
        pos[i] += n * (BALL_RADIUS - d)
        vn = vel[i].dot(n)
        if vn < 0.0:
            vt = vel[i] - vn * n
            vel[i] = vt * (1.0 - FRICTION) - vn * n * RESTITUTION


@ti.func
def ball_collide(i: int, j: int):
    diff = pos[i] - pos[j]
    dsq = diff.dot(diff)
    mind = 2.0 * BALL_RADIUS
    if dsq < mind * mind and dsq > 1e-12:
        d = ti.sqrt(dsq)
        n = diff / d
        ov = mind - d
        pos[i] += 0.5 * ov * n
        pos[j] -= 0.5 * ov * n
        rv = vel[i] - vel[j]
        vn = rv.dot(n)
        if vn > 0.0:
            imp = vn * n
            vel[i] -= 0.5 * imp
            vel[j] += 0.5 * imp


@ti.kernel
def step():
    n = active_count[None]
    for i in range(n):
        if alive[i]:
            vel[i] += GRAVITY * DT
            pos[i] += vel[i] * DT
            if pos[i].y < -3.5:
                alive[i] = 0

    for i in range(n):
        if alive[i]:
            for k in range(NUM_OBS):
                plane_collide(i, obs_center[k], obs_normal[k])

    for i, j in ti.ndrange(n, n):
        if i < j and alive[i] and alive[j]:
            ball_collide(i, j)


@ti.kernel
def spawn_batch(count: int):
    c = active_count[None]
    for k in range(count):
        i = c + k
        if i < NUM_BALLS:
            alive[i] = 1
            pos[i] = ti.Vector([
                ti.random() * 1.6 - 0.8,
                1.2 + ti.random() * 0.35,
                ti.random() * 0.8 - 0.4,
            ])
            vel[i] = ti.Vector([0.0, 0.0, 0.0])
    active_count[None] = ti.min(c + count, NUM_BALLS)


def main():
    init()
    window = ti.ui.Window("Taichi — Ball Cascade", (1280, 820), vsync=True)
    canvas = window.get_canvas()
    canvas.set_background_color((0.03, 0.03, 0.07))
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(2.0, 0.4, 2.4)
    camera.lookat(0.0, -0.15, 0.0)
    camera.fov(52)

    while window.running:
        if active_count[None] + SPAWN_RATE <= NUM_BALLS:
            spawn_batch(SPAWN_RATE)

        for _ in range(SUBSTEPS):
            step()

        camera.track_user_inputs(
            window, movement_speed=0.04, hold_key=ti.ui.LMB
        )
        scene.set_camera(camera)
        scene.point_light(pos=(2.5, 3.5, 3), color=(1.0, 1.0, 1.0))
        scene.point_light(pos=(-2, 2, -2), color=(0.4, 0.4, 0.6))
        scene.ambient_light((0.22, 0.22, 0.32))
        scene.particles(
            pos, radius=BALL_RADIUS, per_vertex_color=ball_rgb,
            index_count=active_count[None],
        )
        scene.lines(
            obs_vtx, width=0.012, indices=obs_idx,
            color=(0.5, 0.5, 0.62), vertex_count=NUM_OBS * 4,
            index_count=NUM_OBS * 8,
        )
        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    main()
