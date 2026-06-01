import taichi as ti
import math
import numpy as np

ANG = math.radians(30)
H = 0.025          # smoothing radius
H2 = H * H
MASS = 0.01
RHO0 = 1000.0
K = 300.0          # stiffer pressure
MU = 1.0           # more viscosity for cohesion
G = ti.Vector([0.0, -30.0])
DT = 0.0005
SUBSTEPS = 4

ti.init(arch=ti.gpu, random_seed=42)

N_MAX = 2000
SPAWN_INTERVAL = 15
SPAWN_COUNT = 5

p = ti.Vector.field(2, float, N_MAX)
v = ti.Vector.field(2, float, N_MAX)
rho = ti.field(float, N_MAX)
pres = ti.field(float, N_MAX)
alive = ti.field(int, N_MAX)
n_alive = ti.field(int, ())

# ── spatial hash ──
GRID_SIZE = int(1.0 / H) + 1
TABLE_SIZE = N_MAX * 2
cell_start = ti.field(int, GRID_SIZE * GRID_SIZE)
cell_end = ti.field(int, GRID_SIZE * GRID_SIZE)
particle_ids = ti.field(int, TABLE_SIZE)
grid_offset = ti.Vector([1.2, 1.5])

# offsets for neighbor search (3×3 grid)
neighbor_offsets = ti.Vector.field(2, int, 9)

@ti.kernel
def init_offsets():
    for i, j in ti.ndrange(3, 3):
        neighbor_offsets[i*3+j] = ti.Vector([i-1, j-1])
init_offsets()

# ── boundary ──
N_BOUNDARY = 600
bp = ti.Vector.field(2, float, N_BOUNDARY)
bp_cnt = ti.field(int, ())

# ── baffles ──
N_BAF = 7
baf_a = ti.Vector.field(2, float, N_BAF)
baf_b = ti.Vector.field(2, float, N_BAF)
MV = N_BAF * 4; MI = N_BAF * 6
mv = ti.Vector.field(3, float, MV)
mn = ti.Vector.field(3, float, MV)
m_idx = ti.field(int, MI)
mc = ti.Vector.field(3, float, MV)

# 3D sync
p3 = ti.Vector.field(3, float, N_MAX)
c3 = ti.Vector.field(3, float, N_MAX)


def build():
    hw = 0.52; xc = 0.866 * hw

    sa = np.zeros((N_BAF, 2), np.float32)
    sb = np.zeros((N_BAF, 2), np.float32)
    v3 = np.zeros((MV, 3), np.float32)
    n3 = np.zeros((MV, 3), np.float32)
    i3 = np.zeros(MI, np.int32)
    cl = np.zeros((MV, 3), np.float32)

    for k in range(N_BAF):
        y = 0.68 - k * 0.22
        s = 1.0 if k % 2 == 0 else -1.0
        cx = -xc * s
        ang = ANG * s
        c = np.array([cx, y], np.float32)
        perp = np.array([math.cos(ang), -math.sin(ang)], np.float32)
        sa[k] = c - perp * hw
        sb[k] = c + perp * hw

        ha = 0.15
        a = sa[k]; b = sb[k]
        v3[k*4+0] = [a[0], a[1], -ha]
        v3[k*4+1] = [b[0], b[1], -ha]
        v3[k*4+2] = [b[0], b[1],  ha]
        v3[k*4+3] = [a[0], a[1],  ha]
        nd = np.array([0.0, 0.0, 1.0], np.float32)
        n3[k*4+0] = nd; n3[k*4+1] = nd; n3[k*4+2] = nd; n3[k*4+3] = nd
        cl[k*4+0] = [0.55, 0.52, 0.45]
        cl[k*4+1] = [0.55, 0.52, 0.45]
        cl[k*4+2] = [0.50, 0.47, 0.40]
        cl[k*4+3] = [0.50, 0.47, 0.40]
        base = k * 4
        i3[k*6+0] = base+0; i3[k*6+1] = base+1; i3[k*6+2] = base+2
        i3[k*6+3] = base+0; i3[k*6+4] = base+2; i3[k*6+5] = base+3

    baf_a.from_numpy(sa); baf_b.from_numpy(sb)
    mv.from_numpy(v3); mn.from_numpy(n3)
    m_idx.from_numpy(i3); mc.from_numpy(cl)

    # boundary particles
    pts = []
    sp = 0.025
    for y in np.arange(-1.2, 1.2, sp):
        pts.append([-1.15, y])
        pts.append([1.15, y])
    for k in range(N_BAF):
        a = sa[k]; b = sb[k]; d = b - a
        nrm = np.array([-d[1], d[0]]); nrm = nrm / (np.linalg.norm(nrm) + 1e-8)
        L = np.linalg.norm(d); n_pts = int(L / sp) + 1
        for j in range(n_pts):
            t = j / max(n_pts-1, 1); pt = a + t * d
            pts.append(pt + nrm * 0.012)
            pts.append(pt - nrm * 0.012)
    bpa = np.array(pts[:N_BOUNDARY], np.float32)
    bp.from_numpy(bpa)
    bp_cnt[None] = len(bpa)

build()


@ti.kernel
def init():
    for i in range(N_MAX):
        p[i] = ti.Vector([0.0, -99.0])
        v[i] = ti.Vector([0.0, 0.0])
        alive[i] = 0
        c3[i] = ti.Vector([0.1, 0.35, 0.80])
    n_alive[None] = 0


# ── spatial hash ──
@ti.func
def grid_cell(pos):
    idx = ti.cast((pos + grid_offset) / H, ti.i32)
    return idx.x * GRID_SIZE + idx.y


@ti.kernel
def build_hash():
    n = n_alive[None]
    # clear
    for i in range(GRID_SIZE * GRID_SIZE):
        cell_start[i] = TABLE_SIZE
        cell_end[i] = TABLE_SIZE
    # count
    for i in range(n):
        if alive[i]:
            cell = grid_cell(p[i])
            ti.atomic_add(cell_end[cell], 1)
    # prefix sum
    total = 0
    for i in range(GRID_SIZE * GRID_SIZE):
        cell_start[i] = total
        total += cell_end[i]
        cell_end[i] = cell_start[i]
    # scatter
    for i in range(n):
        if alive[i]:
            cell = grid_cell(p[i])
            dst = ti.atomic_add(cell_end[cell], 1)
            particle_ids[dst] = i


@ti.kernel
def compute_density_pressure():
    n = n_alive[None]
    for pi in range(n):
        if alive[pi]:
            my_cell = grid_cell(p[pi])
            rho_i = 0.0
            for off_idx in ti.static(range(9)):
                off = neighbor_offsets[off_idx]
                cell = my_cell + off.x * GRID_SIZE + off.y
                if 0 <= cell < GRID_SIZE * GRID_SIZE:
                    beg = cell_start[cell]
                    end = cell_end[cell]
                    for idx in range(beg, end):
                        pj = particle_ids[idx]
                        if alive[pj]:
                            r = p[pi] - p[pj]
                            r2 = r.dot(r)
                            if r2 < H2:
                                q = 1.0 - r2 / H2
                                rho_i += MASS * (315.0 / (64.0 * math.pi * H**9)) * (H2 - r2)**3
            rho[pi] = ti.max(rho_i, RHO0 * 0.5)
            pres[pi] = K * (rho[pi] - RHO0)


@ti.kernel
def compute_forces():
    n = n_alive[None]
    for pi in range(n):
        if alive[pi]:
            f_p = ti.Vector([0.0, 0.0])
            f_v = ti.Vector([0.0, 0.0])
            my_cell = grid_cell(p[pi])
            for off_idx in ti.static(range(9)):
                off = neighbor_offsets[off_idx]
                cell = my_cell + off.x * GRID_SIZE + off.y
                if 0 <= cell < GRID_SIZE * GRID_SIZE:
                    beg = cell_start[cell]
                    end = cell_end[cell]
                    for idx in range(beg, end):
                        pj = particle_ids[idx]
                        if alive[pj] and pi != pj:
                            r = p[pi] - p[pj]
                            r2 = r.dot(r)
                            if r2 < H2 and r2 > 1e-12:
                                dist = ti.sqrt(r2)
                                r_dir = r / dist
                                q = 1.0 - dist / H
                                # pressure gradient
                                grad_w = r_dir * (-945.0 / (32.0 * math.pi * H**9)) * q**2 * (H - dist)**2
                                f_p += -MASS * (pres[pi] + pres[pj]) / (2.0 * rho[pj]) * grad_w
                                # viscosity laplacian
                                lap_w = (945.0 / (32.0 * math.pi * H**9)) * (H - dist) * (3*H - 7*dist)
                                v_diff = v[pj] - v[pi]
                                f_v += MU * MASS * v_diff / rho[pj] * lap_w
            v[pi] += (f_p + f_v + G * rho[pi]) / rho[pi] * DT


@ti.kernel
def boundary_and_integrate():
    n = n_alive[None]
    for pi in range(n):
        if alive[pi]:
            fb = ti.Vector([0.0, 0.0])
            for j in range(bp_cnt[None]):
                r = p[pi] - bp[j]
                dist = r.norm()
                if dist < H * 0.5 and dist > 1e-12:
                    r_dir = r / dist
                    s = (H*0.5 - dist) / (H*0.5)
                    fb += r_dir * s * 400.0
                    v[pi] *= (1.0 - 0.3 * s)
            v[pi] += fb * DT
            p[pi] += v[pi] * DT
            if p[pi].y < -1.5 or p[pi].x < -2.0 or p[pi].x > 2.0:
                alive[pi] = 0


@ti.kernel
def spawn(cnt: ti.i32):
    c = n_alive[None]
    for k in range(cnt):
        i = c + k
        if i < N_MAX:
            alive[i] = 1
            # spawn at top center, tight cluster
            p[i] = ti.Vector([ti.random() * 0.08 - 0.04, 1.02 + ti.random() * 0.02])
            v[i] = ti.Vector([0.0, -0.3])
    f = c + cnt
    if f > N_MAX: f = N_MAX
    n_alive[None] = f


@ti.kernel
def sync():
    no = n_alive[None]
    for i in range(no):
        if alive[i]:
            p3[i] = ti.Vector([p[i].x, p[i].y, 0.0])


def main():
    init()
    sync()

    window = ti.ui.Window("SPH Water Cascade", (1100, 800), vsync=True)
    canvas = window.get_canvas()
    canvas.set_background_color((0.02, 0.02, 0.06))
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0.0, 0.0, 3.0)
    camera.lookat(0.0, 0.0, 0.0)
    camera.up(0.0, 1.0, 0.0)

    frame = 0
    while window.running:
        if n_alive[None] + SPAWN_COUNT <= N_MAX and frame % SPAWN_INTERVAL == 0:
            spawn(SPAWN_COUNT)

        for _ in range(SUBSTEPS):
            build_hash()
            compute_density_pressure()
            compute_forces()
            boundary_and_integrate()

        sync()

        scene.set_camera(camera)
        scene.point_light(pos=(0, 2, 5), color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.55))
        scene.particles(
            p3, radius=0.005,
            per_vertex_color=c3,
            index_count=n_alive[None],
        )
        scene.mesh(mv, m_idx, normals=mn, per_vertex_color=mc, two_sided=True)
        canvas.scene(scene)
        window.show()
        frame += 1


if __name__ == "__main__":
    main()
