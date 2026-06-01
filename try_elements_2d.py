"""
taichi-elements 2D 演示 —— 多材料 MPM。

⚠️ 使用前必读：
    1. 这个库没发布到 PyPI。先 clone：
         git clone https://github.com/taichi-dev/taichi_elements ~/code/taichi_elements
    2. 修改下面 ELEMENTS_PATH 指向你的 clone 路径
    3. uv run python try_elements_2d.py

⚠️ 兼容性警告：
    库上次有意义更新在 2022 年，针对 Taichi 0.8.x。我们当前用的 Taichi 1.7.4
    可能在 import 时就报错（如 `ti.var` 已废弃等）。如果挂了，处理方式：
      (a) 在仓库里 grep 出旧 API，逐个改成新 API；或
      (b) 临时降级 Taichi：uv pip install 'taichi==0.8.11'
    这就是我们当初没走这条路的原因。
"""
import os
import sys
import math

# 改成你的 clone 路径
ELEMENTS_PATH = os.path.expanduser("~/code/taichi_elements")
sys.path.insert(0, ELEMENTS_PATH)

import taichi as ti
from engine.mpm_solver import MPMSolver

ti.init(arch=ti.cpu)  # Mac Metal 不支持 Pointer SNode（稀疏网格），降级到 CPU

# ── 1. 创建 2D MPM 求解器 ──
# res 长度决定维度：(128,128) → 2D；(64,64,64) → 3D
mpm = MPMSolver(res=(128, 128), size=1.0)
mpm.set_gravity((0.0, -9.8))

# ── 2. 添加碰撞体 ──
# add_surface_collider 是无限半平面（point + 外法向）
# 这是 elements 的局限：没有"有限长挡板"，做不了 ball_physics.py 那种 7 层级联
# 只能放一块斜坡，水流到斜坡上滑下去
ANG = math.radians(30)
mpm.add_surface_collider(
    point=(0.0, 0.45),
    normal=(math.sin(ANG), math.cos(ANG)),  # 法向指向"上方"
    surface=mpm.surface_slip,                # slip = 水/沙；sticky = 粘住；separate = 弹开
    friction=0.05,
)

# 四面盒子边界（不开会算飞）
mpm.add_bounding_box(unbounded=False)

# ── 3. 添加粒子（多材料）──
# 这是 elements 最大的卖点：水、弹性体、沙、雪在同一个求解器里共存
def setup_scene():
    """把粒子计数清零并重新撒入三种材料。按 R 键会再次调用。"""
    mpm.n_particles[None] = 0  # 计数清零 → 后续 add_cube 从 index 0 开始覆盖

    mpm.add_cube(
        lower_corner=(0.20, 0.80),
        cube_size=(0.18, 0.12),
        material=MPMSolver.material_water,
        velocity=(2.0, -1.0),
    )
    mpm.add_cube(
        lower_corner=(0.62, 0.80),
        cube_size=(0.12, 0.12),
        material=MPMSolver.material_elastic,
        velocity=(-2.0, -1.0),
    )
    mpm.add_cube(
        lower_corner=(0.43, 0.92),
        cube_size=(0.10, 0.06),
        material=MPMSolver.material_sand,
        velocity=(0.0, -1.5),
    )


setup_scene()


# ── 4. 主循环 ──
COLORS = {
    MPMSolver.material_water:   0x4488FF,  # 蓝
    MPMSolver.material_elastic: 0xFF8844,  # 橘
    MPMSolver.material_sand:    0xFFDD33,  # 黄
    MPMSolver.material_snow:    0xEEEEEE,  # 白
}

gui = ti.GUI("taichi-elements 2D — press R to reset", res=640, background_color=0x112F41)
while gui.running:
    # 键盘事件：R 重置场景，ESC 退出
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key == "r" or e.key == "R":
            setup_scene()
        elif e.key == ti.GUI.ESCAPE:
            gui.running = False

    # 注意：step() 传的是 frame_dt（整帧时长），求解器内部自动子步进
    mpm.step(2e-3)

    info = mpm.particle_info()
    pos = info["position"]       # (N, 2) numpy
    mat = info["material"]       # (N,) numpy int

    # 按材料分组上色（gui.circles 不支持 per-particle 颜色）
    for m, c in COLORS.items():
        mask = mat == m
        if mask.any():
            gui.circles(pos[mask], radius=1.8, color=c)

    # 顺手画一下碰撞斜坡的位置（参考线）
    # 法向 (sin30, cos30) 过 (0, 0.45) → 直线 y = 0.45 - tan(30°) * x
    x0, x1 = 0.0, 1.0
    y0 = 0.45 - math.tan(ANG) * x0
    y1 = 0.45 - math.tan(ANG) * x1
    gui.line([x0, y0], [x1, y1], radius=2, color=0xAAAAAA)

    # 操作提示
    gui.text("R = reset  |  ESC = quit", pos=(0.02, 0.97), color=0xCCCCCC)

    gui.show()
