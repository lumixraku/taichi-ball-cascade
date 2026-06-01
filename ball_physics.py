import taichi as ti
import math
import numpy as np

ANG = math.radians(30)
_S = math.sin(ANG)
_C = math.cos(ANG)

ti.init(arch=ti.gpu, random_seed=42)

N_BALL = 300
N_SPAWN = 0
R = 0.035
G = ti.Vector([0.0, -9.81])
DT = 0.0006
SUB = 12
REST = 0.1
FRIC = 0.05

p2 = ti.Vector.field(2, float, N_BALL)
v2 = ti.Vector.field(2, float, N_BALL)
alive = ti.field(int, N_BALL)
n_alive = ti.field(int, ())
col2 = ti.Vector.field(3, float, N_BALL)
p3 = ti.Vector.field(3, float, N_BALL)

# ── 7 baffles only (for mesh rendering) ──
N_BAF = 7
baf_a = ti.Vector.field(2, float, N_BAF)
baf_b = ti.Vector.field(2, float, N_BAF)

MV = N_BAF * 4; MI = N_BAF * 6
mv = ti.Vector.field(3, float, MV)
mn = ti.Vector.field(3, float, MV)
mi = ti.field(int, MI)
mc = ti.Vector.field(3, float, MV)

# ── all obstacles for collision (baffles + invisible walls, NO floor) ──
N_COL = 9
col_a = ti.Vector.field(2, float, N_COL)
col_b = ti.Vector.field(2, float, N_COL)


def build():
    wa = np.zeros(N_BAF, np.float32)
    ha = np.zeros(N_BAF, np.float32)

    # waterfall cascade: edges meet at x≈0
    # with 30°, perp_x = cos(30°) = 0.866
    # x_center ± 0.866*hw = 0  →  x_center ≈ ±0.866*hw
    hw = 0.52
    xc = 0.866 * hw  # ≈ 0.45 — edges meet at center

    for k in range(N_BAF):
        y = 0.68 - k * 0.22
        s = 1.0 if k % 2 == 0 else -1.0
        cx = -xc * s
        ang = ANG * s
        wa[k] = hw
        ha[k] = 0.15

        # segment endpoints: center ± perpendicular * half_width
        c = np.array([cx, y], np.float32)
        perp = np.array([math.cos(ang), -math.sin(ang)], np.float32)
        baf_a_np = c - perp * wa[k]
        baf_b_np = c + perp * wa[k]

        # write to both baffle field and collision field
        baf_a[k] = baf_a_np
        baf_b[k] = baf_b_np
        col_a[k] = baf_a_np
        col_b[k] = baf_b_np

    # invisible collision walls (no floor — balls fall off screen)
    col_a[7] = [-1.15, -1.2]; col_b[7] = [-1.15, 1.2]   # left wall
    col_a[8] = [1.15, -1.2];  col_b[8] = [1.15, 1.2]    # right wall

    # 3D mesh for baffles only
    v3 = np.zeros((MV, 3), np.float32)
    n3 = np.zeros((MV, 3), np.float32)
    i3 = np.zeros(MI, np.int32)
    c3 = np.zeros((MV, 3), np.float32)

    for i in range(N_BAF):
        a = np.array([baf_a[i].x, baf_a[i].y])
        b = np.array([baf_b[i].x, baf_b[i].y])
        v3[i*4+0] = [a[0], a[1], -ha[i]]
        v3[i*4+1] = [b[0], b[1], -ha[i]]
        v3[i*4+2] = [b[0], b[1],  ha[i]]
        v3[i*4+3] = [a[0], a[1],  ha[i]]
        n3d = np.array([0.0, 0.0, 1.0], np.float32)
        n3[i*4+0] = n3d; n3[i*4+1] = n3d
        n3[i*4+2] = n3d; n3[i*4+3] = n3d
        c3[i*4+0] = [0.92, 0.87, 0.73]
        c3[i*4+1] = [0.92, 0.87, 0.73]
        c3[i*4+2] = [0.88, 0.83, 0.70]
        c3[i*4+3] = [0.88, 0.83, 0.70]
        b = i * 4
        i3[i*6+0] = b+0; i3[i*6+1] = b+1; i3[i*6+2] = b+2
        i3[i*6+3] = b+0; i3[i*6+4] = b+2; i3[i*6+5] = b+3

    mv.from_numpy(v3); mn.from_numpy(n3)
    mi.from_numpy(i3); mc.from_numpy(c3)

build()


@ti.kernel
def init():
    for i in range(N_BALL):
        p2[i] = ti.Vector([0.0, -99.0])
        v2[i] = ti.Vector([0.0, 0.0])
        alive[i] = 0
        rv = ti.random()
        if rv < 0.2:    col2[i] = ti.Vector([0.92, 0.22, 0.12])
        elif rv < 0.4:  col2[i] = ti.Vector([0.95, 0.55, 0.12])
        elif rv < 0.6:  col2[i] = ti.Vector([0.18, 0.58, 0.92])
        elif rv < 0.8:  col2[i] = ti.Vector([0.90, 0.75, 0.14])
        else:           col2[i] = ti.Vector([0.22, 0.72, 0.42])

    for i in range(N_SPAWN):
        alive[i] = 1
        p2[i] = ti.Vector([ti.random() * 1.6 - 0.8, 0.90 + ti.random() * 0.40])
        v2[i] = ti.Vector([ti.random() * 0.4 - 0.2, -0.3])
    n_alive[None] = N_SPAWN


@ti.func
def collide_seg(i: ti.i32, a: ti.template(), b: ti.template()):
    ab = b - a
    ab2 = ab.dot(ab)
    t = (p2[i] - a).dot(ab) / (ab2 + 1e-12)
    if t < 0.0:
        t = 0.0
    if t > 1.0:
        t = 1.0
    closest = a + t * ab
    diff = p2[i] - closest
    dsq = diff.dot(diff)
    if dsq < R * R and dsq > 1e-12:
        d = ti.sqrt(dsq)
        n = diff / d
        p2[i] = closest + n * R
        vn = v2[i].dot(n)
        if vn < 0.0:
            j_n = -(1.0 + REST) * vn
            vt = v2[i] - vn * n
            vts = vt.dot(vt)
            if vts > 1e-12:
                vtm = ti.sqrt(vts)
                vtd = vt / vtm
                jt = vtm
                fl = FRIC * j_n
                if fl < jt:
                    jt = fl
                v2[i] = v2[i] + j_n * n - jt * vtd
            else:
                v2[i] = v2[i] + j_n * n


@ti.func
def collide_ball(i: ti.i32, j: ti.i32):
    diff = p2[i] - p2[j]
    dsq = diff.dot(diff)
    mind = 2.0 * R
    if dsq < mind * mind and dsq > 1e-12:
        d = ti.sqrt(dsq)
        n = diff / d
        ov = mind - d
        p2[i] = p2[i] + 0.5 * ov * n
        p2[j] = p2[j] - 0.5 * ov * n
        rv = v2[i] - v2[j]
        vn = rv.dot(n)
        if vn > 0.0:
            imp = vn * n
            v2[i] = v2[i] - 0.5 * imp
            v2[j] = v2[j] + 0.5 * imp


@ti.kernel
def step():
    n = n_alive[None]
    for i in range(n):
        if alive[i]:
            v2[i] = v2[i] + G * DT
            p2[i] = p2[i] + v2[i] * DT
            if p2[i].y < -1.5:
                alive[i] = 0

    for i in range(n):
        if alive[i]:
            for k in range(N_COL):
                collide_seg(i, col_a[k], col_b[k])

    for i, j in ti.ndrange(n, n):
        if i < j and alive[i] and alive[j]:
            collide_ball(i, j)


@ti.kernel
def spawn(cnt: ti.i32):
    c = n_alive[None]
    for k in range(cnt):
        i = c + k
        if i < N_BALL:
            alive[i] = 1
            p2[i] = ti.Vector([ti.random() * 1.6 - 0.8, 1.0 + ti.random() * 0.25])
            v2[i] = ti.Vector([ti.random() * 0.6 - 0.3, -0.3])
    f = c + cnt
    if f > N_BALL: f = N_BALL
    n_alive[None] = f


@ti.kernel
def sync():
    no = n_alive[None]
    for i in range(no):
        if alive[i]:
            p3[i] = ti.Vector([p2[i].x, p2[i].y, 0.0])


def main():
    init()
    sync()

    window = ti.ui.Window("Ball Cascade — 30°", (1100, 800), vsync=True)
    canvas = window.get_canvas()
    canvas.set_background_color((0.05, 0.05, 0.12))
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0.0, 0.0, 3.0)
    camera.lookat(0.0, 0.0, 0.0)
    camera.up(0.0, 1.0, 0.0)

    frame = 0
    while window.running:
        # ~5 balls per second at 60fps
        if n_alive[None] + 1 <= N_BALL and frame % 60 == 0:
            spawn(1)

        for _ in range(SUB):
            step()

        sync()

        scene.set_camera(camera)
        scene.point_light(pos=(0, 2, 5), color=(1, 1, 1))
        scene.ambient_light((0.45, 0.45, 0.55))

        scene.particles(
            p3, radius=R,
            per_vertex_color=col2,
            index_count=n_alive[None],
        )
        # only baffles rendered as mesh (no ceiling, no floor, no walls)
        scene.mesh(
            mv, mi,
            normals=mn,
            per_vertex_color=mc,
            two_sided=True,
        )
        canvas.scene(scene)
        window.show()
        frame += 1


if __name__ == "__main__":
    main()
