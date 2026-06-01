"""
water_cascade.py 的 taichi-elements 版本。

设计取舍：
  1. elements 只有"无限半平面"碰撞体（add_surface_collider），做不出 7 层有限挡板。
     workaround: 用 material_stationary 沿挡板线段密集摆粒子，让它们在 MPM 网格层面充当固体障碍
  2. elements 默认 E = 1e6 → 真水级的不可压缩。我们手写版用 E=400 → 浆糊
  3. unbounded=True：让粒子飞出 [0,1] 边界自动消失，模拟"流出屏幕"

要求：先 clone taichi_elements 到 ~/code/taichi_elements
"""
import os
import sys
import math
import time

ELEMENTS_PATH = os.path.expanduser("~/code/taichi_elements")
sys.path.insert(0, ELEMENTS_PATH)

import taichi as ti
import numpy as np
from engine.mpm_solver import MPMSolver

ti.init(arch=ti.cpu)  # Mac Metal 不支持 Pointer SNode

# ── MPM 求解器 ──
mpm = MPMSolver(
    res=(96, 96),       # 比 128² 少 44% 网格 → P2G/G2P/collider 都直接对应提速
    size=1.0,
    E_scale=1.0,        # 1.0 = 真水般不可压缩；想要"浆糊"调到 0.05
    unbounded=True,     # 关键：粒子飞出 [0,1] 后自动丢弃 → 模拟"流出屏幕"
    use_g2p2g=True,     # 融合 G2P+P2G 的优化算法，~25% 提速
)
mpm.set_gravity((0.0, -40.0))   # 加大重力让水流得快、像瀑布

# ── 挡板几何（同 ball_physics.py 的布局）──
ANG = math.radians(30)
N_BAF = 7
HW = 0.24       # 半宽
Y_TOP = 0.85
Y_SPACING = 0.11
XC = 0.225

def build_baffles():
    """挡板：物理上用 surface_slip 球链（不可见），视觉上用线段。
    返回 (spheres, line_endpoints)：
      spheres: [(cx, cy, r), ...]  → collider 注册
      line_endpoints: [(ax, ay, bx, by), ...]  → gui.line 渲染
    """
    spheres = []
    lines = []
    SPH_R = 0.040           # 更大半径 → 更少 collider；2r=0.08 > spacing=0.06 不漏
    SPH_SPACING = 0.060
    for k in range(N_BAF):
        y = Y_TOP - k * Y_SPACING
        s = 1.0 if k % 2 == 0 else -1.0
        cx = 0.5 - XC * s
        ang = ANG * s
        d = np.array([math.cos(ang), -math.sin(ang)])
        center = np.array([cx, y])
        a = center - d * HW
        b = center + d * HW
        lines.append((float(a[0]), float(a[1]), float(b[0]), float(b[1])))
        L = HW * 2
        n_along = max(int(L / SPH_SPACING) + 1, 2)
        for i in range(n_along):
            t = i / (n_along - 1)
            pt = a + t * (b - a)
            spheres.append((float(pt[0]), float(pt[1]), SPH_R))
    return spheres, lines


BAFFLE_SPHERES, BAFFLE_LINES = build_baffles()
print(f"baffle sphere colliders: {len(BAFFLE_SPHERES)}  |  lines: {len(BAFFLE_LINES)}")


# ── 左右隐形墙（无限半平面没问题，因为左右就是要 ±∞ 高）──
# 不加上下：上方留口让水流入，下方留口让水流出
def add_walls():
    mpm.add_surface_collider(
        point=(0.04, 0.0), normal=(1.0, 0.0),
        surface=mpm.surface_slip, friction=0.0,
    )
    mpm.add_surface_collider(
        point=(0.96, 0.0), normal=(-1.0, 0.0),
        surface=mpm.surface_slip, friction=0.0,
    )


# ── 批量球碰撞体（关键性能优化）──
# elements 的 add_sphere_collider 会为每个球注册一个 ti.kernel；
# 91 个球 × ~1000 内部子步 = ~9 万次 kernel 启动/渲染帧，CPU 受不了。
# 这里把所有球的 SDF 检查塞进一个 batched kernel，启动次数 → 1。
N_SPH = len(BAFFLE_SPHERES)
sph_centers = ti.Vector.field(2, ti.f32, shape=N_SPH)
sph_radii = ti.field(ti.f32, shape=N_SPH)
sph_centers.from_numpy(np.array([[s[0], s[1]] for s in BAFFLE_SPHERES], dtype=np.float32))
sph_radii.from_numpy(np.array([s[2] for s in BAFFLE_SPHERES], dtype=np.float32))

DX = mpm.dx

@ti.kernel
def batched_sphere_collide(t: ti.f32, dt: ti.f32, grid_v: ti.template()):
    for I in ti.grouped(grid_v):
        v = grid_v[I]
        cell_pos = I.cast(ti.f32) * DX
        for k in range(N_SPH):
            offset = cell_pos - sph_centers[k]
            r = sph_radii[k]
            if offset.norm_sqr() < r * r:
                n = offset.normalized(1e-5)
                v = v - n * n.dot(v)   # surface_slip: 去掉法向分量
        grid_v[I] = v


def add_baffles():
    """把批量 collider 注册到求解器（只调用一次）。"""
    mpm.grid_postprocess.append(batched_sphere_collide)


def spawn_water_drop():
    """顶部喷一坨水。"""
    mpm.add_cube(
        lower_corner=(0.46, 0.93),
        cube_size=(0.10, 0.04),
        material=MPMSolver.material_water,
        color=0x4DA6FF,
        velocity=(0.0, -1.5),
    )


# collider 只注册一次（add_sphere_collider 不是 particle，重置不用重加）
add_walls()
add_baffles()
spawn_water_drop()


# ── 主循环 ──
COLORS = {
    MPMSolver.material_water: np.uint32(0x4DA6FF),
}

gui = ti.GUI("water cascade (elements) — R reset, ESC quit",
             res=720, background_color=0x081428)
SPAWN_INTERVAL_SEC = 5.0   # 每 5 秒（墙钟）喷一坨水
last_spawn = time.time()

while gui.running:
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key in ("r", "R"):
            mpm.n_particles[None] = 0   # collider 不动，只清粒子
            spawn_water_drop()
            last_spawn = time.time()
        elif e.key == ti.GUI.ESCAPE:
            gui.running = False

    now = time.time()
    if now - last_spawn >= SPAWN_INTERVAL_SEC:
        spawn_water_drop()
        last_spawn = now

    mpm.step(8e-3)   # 加大每帧模拟时长 → 视觉上流得更快

    info = mpm.particle_info()
    pos = info["position"]
    mat = info["material"]

    # unbounded 模式下，逸出 [0,1] 的粒子坐标也会一直增长。
    # 简单 mask 掉就行——它们在物理上已"失踪"
    in_view = (pos[:, 0] > 0.0) & (pos[:, 0] < 1.0) & \
              (pos[:, 1] > 0.0) & (pos[:, 1] < 1.0)

    # 先画挡板（线段渲染，物理上是底下的球链 collider）
    for ax, ay, bx, by in BAFFLE_LINES:
        gui.line([ax, ay], [bx, by], radius=3, color=0xB59E6D)

    # 再画水
    for m, c in COLORS.items():
        mask = (mat == m) & in_view
        if mask.any():
            gui.circles(pos[mask], radius=2.0, color=int(c))

    gui.text(f"R = reset  |  ESC = quit  |  particles = {int(mpm.n_particles[None])}",
             pos=(0.02, 0.97), color=0xCCCCCC)
    gui.show()
