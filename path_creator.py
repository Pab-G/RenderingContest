import json
import numpy as np
import math

# Load the user‑provided JSON to copy intrinsics & metadata
input_path = "./data/nerf_synthetic/nubzuki_only_v2/transforms_test.json"
with open(input_path, "r") as f:
    data = json.load(f)

# Keep all keys except we will overwrite "frames"
output = {k: v for k, v in data.items() if k != "frames"}

# Convenience constants
CENTER = np.array([0.0, 0.0, -0.5])
RADIUS = 2.0
UP = np.array([0.0, 0.0, 1.0])

frames = []

def make_rotation(forward):
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, UP)
    right /= np.linalg.norm(right)
    up = UP  # keep global up
    # Columns: right, up, -forward  (matches user's file)
    R = np.column_stack([right, up, -forward])
    return R

def add_frame(pos, forward, idx):
    R = make_rotation(forward)
    tx, ty, tz = pos
    mat = [
        [float(R[0,0]), float(R[0,1]), float(R[0,2]), float(tx)],
        [float(R[1,0]), float(R[1,1]), float(R[1,2]), float(ty)],
        [float(R[2,0]), float(R[2,1]), float(R[2,2]), float(tz)],
        [0.0, 0.0, 0.0, 1.0]
    ]
    frames.append({
        "file_path": f"images/circle_{idx:05d}.jpg",
        "transform_matrix": mat
    })

# ---- Segment 1 : 80 cameras, full 360° around centre ----
print("first part")
num1 = 80
angles1 = np.linspace(0, 2*np.pi, num1, endpoint=False)
for i, theta in enumerate(angles1):
    x = CENTER[0] + RADIUS * math.cos(theta)
    y = CENTER[1] + RADIUS * math.sin(theta)
    z = CENTER[2]
    pos = np.array([x, y, z])
    forward = CENTER - pos
    add_frame(pos, forward, len(frames))

# ---- Segment 2 : 50 cameras, ~200° arc (bit more than half turn) ----
print("second part")
num2 = 70
arc2 = 1.25 * math.pi           


t = np.linspace(0, 1, num2)
ease_out_t = 1 - (1 - t)**2
angles2 = 0 + ease_out_t * arc2

for theta in angles2:
    x = CENTER[0] + RADIUS * math.cos(theta)
    y = CENTER[1] + RADIUS * math.sin(theta)
    z = CENTER[2]
    pos = np.array([x, y, z])
    forward = CENTER - pos
    add_frame(pos, forward, len(frames))

# ---- Segment 3 : 20 cameras along a smooth circular arc to (0,0.5,-0.5) ----
num3 = 20
end_angle = np.deg2rad(90)
start_angle = np.deg2rad(135)
angles3 = np.linspace(start_angle, end_angle, num3, endpoint=False)
CENTER = np.array([0.0, -2.8284271, -0.5])
RADIUS = 1.8284271

for i, theta in enumerate(angles3):
    x = CENTER[0] + RADIUS * math.cos(theta)
    y = CENTER[1] + RADIUS * math.sin(theta)
    z = CENTER[2]
    pos = np.array([x, y, z])
    
    tangent = np.array([math.sin(theta), -math.cos(theta), 0])
    
    add_frame(pos, tangent, len(frames))

num4 = 30
end = 1.5 * 2 + 1.0
t = np.linspace(0, end, num4)
for i,n in enumerate(t):
    x = (n + 1.5/2) % 1.5 - 1.5/2
    y = -1.0
    z = -0.5
    pos = np.array([x, y, z])
    
    tangent = np.array([1.0,0.0,0.0])
    
    add_frame(pos, tangent, len(frames))

num4 = 30
arc4 = 0.5 * math.pi 
start_angle4 = 1.5 * math.pi 
t = np.linspace(0, 1, num4)
ease_out_t = 1 - (1 - t)**2
angles4 = start_angle4 + t * arc4
CENTER = np.array([-2, -2, -0.5])
RADIUS = 2.0

for i, theta in enumerate(angles4):
    x = CENTER[0] + RADIUS * math.cos(theta)
    y = CENTER[1] + RADIUS * math.sin(theta)
    z = CENTER[2]
    pos = np.array([x, y, z])
    
    tangent = np.array([-math.sin(theta), math.cos(theta), 0])
    
    add_frame(pos, tangent, len(frames))

num5 = 30
arc5 = 0.5 * math.pi 
start_angle5 = 1.5 * math.pi 
t = np.linspace(0, 1, num5)
ease_out_t = 1 - (1 - t)**2
angles5 = start_angle5 + ease_out_t * arc5
CENTER = np.array([0.0, 0.0, -0.5])
RADIUS = 2.0

for i, theta in enumerate(angles5):
    x = CENTER[0] + RADIUS * math.cos(theta)
    y = CENTER[1] + RADIUS * math.sin(theta)
    z = CENTER[2]
    pos = np.array([x, y, z])
    forward = CENTER - pos
    add_frame(pos, forward, len(frames))

assert len(frames) == 260, f"Total frames {len(frames)}≠260"

output["frames"] = frames

out_path = "./data/nerf_synthetic/nubzuki_only_v2/transforms_test.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=4)

out_path
