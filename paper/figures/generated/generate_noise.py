import numpy as np
from PIL import Image, ImageDraw

np.random.seed(42)

Nx, Ny, Nz = 8, 5, 5
cell = 80
res = 10

# Screen coords: x right, y down
# Shallower angle (15°) so front face is more prominent
# Dimetric projection: front face prominent, good top visibility
# x-axis: right, slight down
# y-axis (depth): right-up, foreshortened
# z-axis: straight up
ix = np.array([np.cos(np.radians(15)), np.sin(np.radians(15))]) * cell
iy = np.array([np.cos(np.radians(30)), -np.sin(np.radians(30))]) * cell
iz = np.array([0, -1]) * cell

margin = 10
corners = []
for i in [0, Nx]:
    for j in [0, Ny]:
        for k in [0, Nz]:
            corners.append(i * ix + j * iy + k * iz)
xs = [c[0] for c in corners]
ys = [c[1] for c in corners]
ox = -min(xs) + margin
oy = -min(ys) + margin
W = int(max(xs) - min(xs) + 2 * margin)
H = int(max(ys) - min(ys) + 2 * margin)

img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

def project(i, j, k):
    p = i * ix + j * iy + k * iz
    return (int(ox + p[0]), int(oy + p[1]))

def lerp(p0, p1, t):
    return (int(p0[0] + (p1[0] - p0[0]) * t), int(p0[1] + (p1[1] - p0[1]) * t))

def noise_texture(mean, std):
    return np.clip(np.random.normal(mean, std, (res, res, 3)), 0, 255).astype(np.uint8)

def fill_quad(corners, tex):
    tl, tr, br, bl = corners
    for row in range(res):
        for col in range(res):
            u0, u1 = col / res, (col + 1) / res
            v0, v1 = row / res, (row + 1) / res
            p00 = lerp(lerp(tl, tr, u0), lerp(bl, br, u0), v0)
            p10 = lerp(lerp(tl, tr, u1), lerp(bl, br, u1), v0)
            p11 = lerp(lerp(tl, tr, u1), lerp(bl, br, u1), v1)
            p01 = lerp(lerp(tl, tr, u0), lerp(bl, br, u0), v1)
            color = tuple(tex[row, col])
            draw.polygon([p00, p10, p11, p01], fill=color)

def draw_grid(face_fn, Nu, Nv, width):
    for v in range(Nv + 1):
        for u in range(Nu + 1):
            if u < Nu:
                draw.line([face_fn(u, v), face_fn(u + 1, v)], fill=(0, 0, 0), width=width)
            if v < Nv:
                draw.line([face_fn(u, v), face_fn(u, v + 1)], fill=(0, 0, 0), width=width)

# Visible faces from standard isometric viewpoint: front (y=Ny), right (x=Nx), top (z=Nz)
# Draw back-to-front

# Draw back-to-front: top first, then right, then front (y=0)

# Top face (z=Nz plane, spans x and y)
for j in range(Ny - 1, -1, -1):
    for i in range(Nx):
        bl = project(i, j, Nz)
        br = project(i + 1, j, Nz)
        tr = project(i + 1, j + 1, Nz)
        tl = project(i, j + 1, Nz)
        fill_quad((tl, tr, br, bl), noise_texture(175, 35))

# Right face (x=Nx plane, spans y and z)
for j in range(Ny - 1, -1, -1):
    for k in range(Nz):
        tl = project(Nx, j, k + 1)
        tr = project(Nx, j + 1, k + 1)
        br = project(Nx, j + 1, k)
        bl = project(Nx, j, k)
        fill_quad((tl, tr, br, bl), noise_texture(160, 35))

# Front face (y=0 plane, spans x and z)
for i in range(Nx):
    for k in range(Nz):
        tl = project(i, 0, k + 1)
        tr = project(i + 1, 0, k + 1)
        br = project(i + 1, 0, k)
        bl = project(i, 0, k)
        fill_quad((tl, tr, br, bl), noise_texture(190, 45))

# Grid lines — top
draw_grid(lambda u, v: project(u, v, Nz), Nx, Ny, 6)
# Grid lines — right
draw_grid(lambda u, v: project(Nx, u, v), Ny, Nz, 6)
# Grid lines — front (y=0)
draw_grid(lambda u, v: project(u, 0, v), Nx, Nz, 6)

# Bold outer edges
outer = [
    # Front face outline (y=0)
    (project(0, 0, 0), project(Nx, 0, 0)),
    (project(Nx, 0, 0), project(Nx, 0, Nz)),
    (project(Nx, 0, Nz), project(0, 0, Nz)),
    (project(0, 0, Nz), project(0, 0, 0)),
    # Top face back edges
    (project(0, 0, Nz), project(0, Ny, Nz)),
    (project(0, Ny, Nz), project(Nx, Ny, Nz)),
    (project(Nx, Ny, Nz), project(Nx, 0, Nz)),
    # Right face back edges
    (project(Nx, 0, 0), project(Nx, Ny, 0)),
    (project(Nx, Ny, 0), project(Nx, Ny, Nz)),
]
for p0, p1 in outer:
    draw.line([p0, p1], fill=(0, 0, 0), width=10)

img.save("paper/figures/generated/noise_cube.png")
print("Saved noise_cube.png")
