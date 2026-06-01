import taichi as ti

ti.init(arch=ti.gpu)

n_particles = 8192
n_grid = 128
dx = 1.0 / n_grid
inv_dx = float(n_grid)
dt = 2e-4
substeps = 25

p_rho = 1.0
p_vol = (dx * 0.5) ** 2
p_mass = p_vol * p_rho
gravity = 9.8
E = 400.0  # 弱可压缩水的体积模量；越大越"硬"

x = ti.Vector.field(2, float, n_particles)   # 位置
v = ti.Vector.field(2, float, n_particles)   # 速度
C = ti.Matrix.field(2, 2, float, n_particles)  # APIC 仿射速度场
J = ti.field(float, n_particles)             # 体积比 (det F)，水只追这个标量

grid_v = ti.Vector.field(2, float, (n_grid, n_grid))
grid_m = ti.field(float, (n_grid, n_grid))


@ti.kernel
def init():
    for i in range(n_particles):
        x[i] = [ti.random() * 0.3 + 0.35, ti.random() * 0.25 + 0.7]
        v[i] = [0.0, -1.0]
        J[i] = 1.0
        C[i] = ti.Matrix.zero(float, 2, 2)


@ti.kernel
def substep():
    # 清网格
    for i, j in grid_m:
        grid_v[i, j] = [0.0, 0.0]
        grid_m[i, j] = 0.0

    # P2G：粒子 → 网格
    for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2,
             0.75 - (fx - 1.0) ** 2,
             0.5 * (fx - 0.5) ** 2]
        # 水的应力：只跟体积比 J 有关
        stress = -dt * 4.0 * E * p_vol * (J[p] - 1.0) * inv_dx * inv_dx
        affine = ti.Matrix([[stress, 0.0], [0.0, stress]]) + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i].x * w[j].y
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass

    # 网格更新：重力 + 边界
    for i, j in grid_m:
        if grid_m[i, j] > 0.0:
            grid_v[i, j] /= grid_m[i, j]
            grid_v[i, j].y -= dt * gravity
            if i < 3 and grid_v[i, j].x < 0: grid_v[i, j].x = 0
            if i > n_grid - 3 and grid_v[i, j].x > 0: grid_v[i, j].x = 0
            if j < 3 and grid_v[i, j].y < 0: grid_v[i, j].y = 0
            if j > n_grid - 3 and grid_v[i, j].y > 0: grid_v[i, j].y = 0

    # G2P：网格 → 粒子
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
        x[p] += dt * new_v
        J[p] *= 1.0 + dt * new_C.trace()
        C[p] = new_C


def main():
    init()
    gui = ti.GUI("MPM Water (2D)", res=600, background_color=0x061224)
    while gui.running:
        for _ in range(substeps):
            substep()
        gui.circles(x.to_numpy(), radius=1.5, color=0x4D8FE3)
        gui.show()


if __name__ == "__main__":
    main()
