import taichi as ti
import math
import numpy as np

# ── 预设 ──
PRESETS = {
    "water":  dict(gravity=15.0, E=400.0,  FRIC=0.985, dt=2.0e-4, substeps=20, color=(0.20, 0.55, 0.92)),
    "honey":  dict(gravity=6.0,  E=300.0,  FRIC=0.90,  dt=2.0e-4, substeps=20, color=(0.95, 0.72, 0.18)),
    "splash": dict(gravity=28.0, E=1200.0, FRIC=0.998, dt=1.2e-4, substeps=25, color=(0.55, 0.88, 0.98)),
    "mud":    dict(gravity=10.0, E=150.0,  FRIC=0.95,  dt=2.5e-4, substeps=18, color=(0.55, 0.40, 0.25)),
}
START_PRESET = "water"

# 运行时 Python 状态（GUI 直接读写）
PARAMS = dict(PRESETS[START_PRESET])

ti.init(arch=ti.gpu, random_seed=42)

# ── 固定参数 ──
n_particles = 10000   # 取所有预设的最大值，多余的粒子也无副作用
n_grid = 128
dx = 1.0 / n_grid
inv_dx = float(n_grid)
p_rho = 1.0
p_vol = (dx * 0.5) ** 2
p_mass = p_vol * p_rho

ANG = math.radians(30)
N_BAF = 7
HW = 0.26
THICK = 2.0 * dx
Y_TOP = 0.84
Y_SPACING = 0.11
XC = 0.225

# ── 字段 ──
x = ti.Vector.field(2, float, n_particles)
v = ti.Vector.field(2, float, n_particles)
C = ti.Matrix.field(2, 2, float, n_particles)
J = ti.field(float, n_particles)
grid_v = ti.Vector.field(2, float, (n_grid, n_grid))
grid_m = ti.field(float, (n_grid, n_grid))
baf_a = ti.Vector.field(2, float, N_BAF)
baf_b = ti.Vector.field(2, float, N_BAF)

# 渲染
p3 = ti.Vector.field(3, float, n_particles)
col = ti.Vector.field(3, float, n_particles)
MV = N_BAF * 4
MI = N_BAF * 6
mv = ti.Vector.field(3, float, MV)
mn = ti.Vector.field(3, float, MV)
mi = ti.field(int, MI)
mc = ti.Vector.field(3, float, MV)

# 运行时可变（kernel 通过 [None] 读取）
rt_gravity = ti.field(float, ())
rt_E       = ti.field(float, ())
rt_FRIC    = ti.field(float, ())
rt_dt      = ti.field(float, ())
rt_color   = ti.Vector.field(3, float, ())


def build_baffles():
    sa = np.zeros((N_BAF, 2), np.float32)
    sb = np.zeros((N_BAF, 2), np.float32)
    v3 = np.zeros((MV, 3), np.float32)
    n3 = np.zeros((MV, 3), np.float32)
    i3 = np.zeros(MI, np.int32)
    c3 = np.zeros((MV, 3), np.float32)
    depth = 0.08

    for k in range(N_BAF):
        y_u = Y_TOP - k * Y_SPACING
        s = 1.0 if k % 2 == 0 else -1.0
        cx_u = 0.5 - XC * s
        ang = ANG * s
        c_u = np.array([cx_u, y_u], np.float32)
        perp = np.array([math.cos(ang), -math.sin(ang)], np.float32)
        sa[k] = c_u - perp * HW
        sb[k] = c_u + perp * HW

        aw = sa[k] * 2.0 - 1.0
        bw = sb[k] * 2.0 - 1.0
        v3[k*4+0] = [aw[0], aw[1], -depth]
        v3[k*4+1] = [bw[0], bw[1], -depth]
        v3[k*4+2] = [bw[0], bw[1],  depth]
        v3[k*4+3] = [aw[0], aw[1],  depth]
        nd = np.array([0.0, 0.0, 1.0], np.float32)
        n3[k*4+0] = nd; n3[k*4+1] = nd
        n3[k*4+2] = nd; n3[k*4+3] = nd
        c3[k*4+0] = [0.92, 0.87, 0.73]
        c3[k*4+1] = [0.92, 0.87, 0.73]
        c3[k*4+2] = [0.88, 0.83, 0.70]
        c3[k*4+3] = [0.88, 0.83, 0.70]
        base = k * 4
        i3[k*6+0] = base+0; i3[k*6+1] = base+1; i3[k*6+2] = base+2
        i3[k*6+3] = base+0; i3[k*6+4] = base+2; i3[k*6+5] = base+3

    baf_a.from_numpy(sa); baf_b.from_numpy(sb)
    mv.from_numpy(v3); mn.from_numpy(n3)
    mi.from_numpy(i3); mc.from_numpy(c3)


build_baffles()


@ti.func
def baffle_clamp(pos, vel):
    new_vel = vel
    for k in ti.static(range(N_BAF)):
        a = baf_a[k]
        b = baf_b[k]
        ab = b - a
        t = (pos - a).dot(ab) / (ab.dot(ab) + 1e-12)
        t = ti.max(0.0, ti.min(1.0, t))
        closest = a + t * ab
        diff = pos - closest
        d = diff.norm()
        if d < THICK and d > 1e-8:
            n = diff / d
            vn = new_vel.dot(n)
            if vn < 0.0:
                tang = new_vel - vn * n
                new_vel = tang * rt_FRIC[None]
    return new_vel


@ti.kernel
def init():
    for i in range(n_particles):
        x[i] = [ti.random() * 0.14 + 0.43, ti.random() * 0.88 + 0.08]
        v[i] = [0.0, -0.3]
        J[i] = 1.0
        C[i] = ti.Matrix.zero(float, 2, 2)
        col[i] = rt_color[None]


@ti.kernel
def repaint():
    for i in range(n_particles):
        col[i] = rt_color[None]


@ti.kernel
def substep():
    _dt = rt_dt[None]
    _g  = rt_gravity[None]
    _E  = rt_E[None]

    for i, j in grid_m:
        grid_v[i, j] = [0.0, 0.0]
        grid_m[i, j] = 0.0

    for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2,
             0.75 - (fx - 1.0) ** 2,
             0.5 * (fx - 0.5) ** 2]
        stress = -_dt * 4.0 * _E * p_vol * (J[p] - 1.0) * inv_dx * inv_dx
        affine = ti.Matrix([[stress, 0.0], [0.0, stress]]) + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i].x * w[j].y
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass

    for i, j in grid_m:
        if grid_m[i, j] > 0.0:
            grid_v[i, j] /= grid_m[i, j]
            grid_v[i, j].y -= _dt * _g
            if i < 3 and grid_v[i, j].x < 0: grid_v[i, j].x = 0
            if i > n_grid - 3 and grid_v[i, j].x > 0: grid_v[i, j].x = 0
            pos = ti.Vector([i * dx, j * dx])
            grid_v[i, j] = baffle_clamp(pos, grid_v[i, j])

    for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2,
             0.75 - (fx - 1.0) ** 2,
             0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            g_v = grid_v[base + offset]
            weight = w[i].x * w[j].y
            new_v += weight * g_v
            new_C += 4.0 * inv_dx * weight * g_v.outer_product(dpos)
        v[p] = new_v
        x[p] += _dt * new_v
        J[p] *= 1.0 + _dt * new_C.trace()
        C[p] = new_C

        if x[p].y < 0.04:
            x[p] = ti.Vector([ti.random() * 0.14 + 0.43, 0.94 + ti.random() * 0.04])
            v[p] = ti.Vector([0.0, -0.3])
            J[p] = 1.0
            C[p] = ti.Matrix.zero(float, 2, 2)


@ti.kernel
def sync_render():
    for i in range(n_particles):
        p3[i] = ti.Vector([x[i].x * 2.0 - 1.0, x[i].y * 2.0 - 1.0, 0.0])


def push_params():
    rt_gravity[None] = PARAMS["gravity"]
    rt_E[None]       = PARAMS["E"]
    rt_FRIC[None]    = PARAMS["FRIC"]
    rt_dt[None]      = PARAMS["dt"]
    rt_color[None]   = PARAMS["color"]


def apply_preset(name: str):
    PARAMS.update(PRESETS[name])
    push_params()
    repaint()


def main():
    push_params()
    init()
    sync_render()

    window = ti.ui.Window("MPM Cascade", (1100, 800), vsync=True)
    canvas = window.get_canvas()
    canvas.set_background_color((0.03, 0.04, 0.10))
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0.0, 0.0, 3.0)
    camera.lookat(0.0, 0.0, 0.0)
    camera.up(0.0, 1.0, 0.0)
    gui = window.get_gui()

    while window.running:
        # GUI 面板
        with gui.sub_window("Controls", 0.02, 0.02, 0.26, 0.46):
            gui.text("Preset")
            if gui.button("water"):  apply_preset("water")
            if gui.button("honey"):  apply_preset("honey")
            if gui.button("splash"): apply_preset("splash")
            if gui.button("mud"):    apply_preset("mud")
            gui.text("Tweak")
            PARAMS["gravity"]  = gui.slider_float("gravity",  PARAMS["gravity"], 0.0, 40.0)
            PARAMS["E"]        = gui.slider_float("E (stiff)", PARAMS["E"],     50.0, 1500.0)
            PARAMS["FRIC"]     = gui.slider_float("FRIC",     PARAMS["FRIC"],   0.80, 1.00)
            PARAMS["substeps"] = gui.slider_int("substeps",   PARAMS["substeps"], 5, 40)
            if gui.button("reset particles"):
                init()

        push_params()

        for _ in range(PARAMS["substeps"]):
            substep()
        sync_render()

        scene.set_camera(camera)
        scene.point_light(pos=(0, 2, 5), color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.55))
        scene.particles(p3, radius=0.012, per_vertex_color=col)
        scene.mesh(mv, mi, normals=mn, per_vertex_color=mc, two_sided=True)
        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    main()
