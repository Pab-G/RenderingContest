"""
PyTorch implementation of Gaussian Splat Rasterizer.

The implementation is based on torch-splatting: https://github.com/hbb1/torch-splatting
"""

from jaxtyping import Bool, Float, jaxtyped
import torch
import math
from typeguard import typechecked


from .camera import Camera
from .scene import Scene
from .sh import eval_sh




class GSRasterizer(object):
    """
    Gaussian Splat Rasterizer.
    """

    def __init__(self):

        self.sh_degree = 3
        self.white_bkgd = True
        self.tile_size = 25  #36 for nubzuki

    def render_scene(self, scene: Scene, camera: Camera, idx):

        # Retrieve Gaussian parameters
        mean_3d = scene.mean_3d
        scales = scene.scales
        rotations = scene.rotations
        shs = scene.shs
        opacities = scene.opacities
        
        # ============================================================================
        # Process camera parameters
        # NOTE: We transpose both camera extrinsic and projection matrices
        # assuming that these transforms are applied to points in row vector format.
        # NOTE: Do NOT modify this block.
        # Retrieve camera pose (extrinsic)

        z_deg = -130
        y_deg = 0
        x_deg = 0
        x_t = 0
        y_t = 0
        z_t = 0
        w2o = transform(x_deg,y_deg,z_deg,x_t,y_t,z_t,device=mean_3d.device)
        #camera.camera_to_world = torch.matmul(w2o,camera.camera_to_world)

        #camera.camera_to_world = camerafixe(camera.camera_to_world,x_deg,y_deg,z_deg,mean_3d.device)
        mean_3d,scales,rotations,shs,opacities = filterfixe(mean_3d,scales,rotations,shs,opacities)
        mean_3d,scales,rotations,shs,opacities = alignz(mean_3d,scales,rotations,shs,opacities,130.0)
        mean_3d,scales,rotations,shs,opacities = idx_manager(mean_3d,scales,rotations,shs,opacities,idx)
        mean_3d,scales,rotations,shs,opacities = filter(camera.camera_to_world,mean_3d,scales,rotations,shs,opacities)

        R = camera.camera_to_world[:3, :3]  # 3 x 3
        T = camera.camera_to_world[:3, 3:4]  # 3 x 1
        R_edit = torch.diag(torch.tensor([1, -1, -1], device=R.device, dtype=R.dtype))
        R = R @ R_edit
        R_inv = R.T
        T_inv = -R_inv @ T
        world_to_camera = torch.eye(4, device=R.device, dtype=R.dtype)
        world_to_camera[:3, :3] = R_inv
        world_to_camera[:3, 3:4] = T_inv
        world_to_camera = world_to_camera.permute(1, 0)

        # Retrieve camera intrinsic
        proj_mat = camera.proj_mat.permute(1, 0)
        world_to_camera = world_to_camera.to(mean_3d.device)
        proj_mat = proj_mat.to(mean_3d.device)
        # ============================================================================

        # Project Gaussian center positions to NDC
        mean_ndc, mean_view, in_mask = self.project_ndc(
            mean_3d, world_to_camera, proj_mat, camera.near,
        )
        mean_ndc = mean_ndc[in_mask]
        mean_view = mean_view[in_mask]
        mean_3d    = mean_3d   [in_mask]
        scales     = scales    [in_mask]
        rotations  = rotations [in_mask]
        shs        = shs       [in_mask]
        opacities  = opacities [in_mask]
        assert mean_ndc.shape[0] > 0, "No points in the frustum"
        assert mean_view.shape[0] > 0, "No points in the frustum"
        depths = mean_view[:, 2]

        # Compute RGB from spherical harmonics
        color = self.get_rgb_from_sh(mean_3d, shs, camera)

        # Compute 3D covariance matrix
        cov_3d = self.compute_cov_3d(scales, rotations)

        # Project covariance matrices to 2D
        cov_2d = self.compute_cov_2d(
            mean_3d=mean_3d, 
            cov_3d=cov_3d, 
            w2c=world_to_camera,
            f_x=camera.f_x, 
            f_y=camera.f_y,
        )
        
        # Compute pixel space coordinates of the projected Gaussian centers
        mean_coord_x = ((mean_ndc[..., 0] + 1) * camera.image_width - 1.0) * 0.5
        mean_coord_y = ((mean_ndc[..., 1] + 1) * camera.image_height - 1.0) * 0.5
        mean_2d = torch.stack([mean_coord_x, mean_coord_y], dim=-1)

        color = self.render(
            camera=camera, 
            mean_2d=mean_2d,
            cov_2d=cov_2d,
            color=color,
            opacities=opacities, 
            depths=depths,
        )
        color = color.reshape(-1, 3)

        return color

    @torch.no_grad()
    def get_rgb_from_sh(self, mean_3d, shs, camera):
        rays_o = camera.cam_center        
        rays_d = mean_3d - rays_o
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        color = eval_sh(self.sh_degree, shs.permute(0, 2, 1), rays_d)
        color = torch.clamp_min(color + 0.5, 0.0)
        return color
    
    @jaxtyped(typechecker=typechecked)
    @torch.no_grad()
    def project_ndc(
        self,
        points: Float[torch.Tensor, "N 3"],
        w2c: Float[torch.Tensor, "4 4"],
        proj_mat: Float[torch.Tensor, "4 4"],
        z_near: float,
    ) -> tuple[
        Float[torch.Tensor, "N 4"],
        Float[torch.Tensor, "N 4"],
        Bool[torch.Tensor, "N"],
    ]:
        """
        Projects points to NDC space.
        
        Args:
        - points: 3D points in object space.
        - w2c: World-to-camera matrix.
        - proj_mat: Projection matrix.
        - z_near: Near plane distance.

        Returns:
        - p_ndc: NDC coordinates.
        - p_view: View space coordinates.
        - in_mask: Mask of points that are in the frustum.
        """
        # ========================================================
        # TODO: Implement the projection to NDC space
        p_view = homogenize(points) @ w2c  # Apply world-to-camera transformation
        p_proj = p_view @ proj_mat
        p_ndc = p_proj / (p_proj[:, 3:4]) #.clamp(min=1e-6)

        # TODO: Cull points that are close or behind the camera
        in_mask = (p_view[:, 2] > z_near) #& (p_ndc[:, 2] < p_ndc[:, 3])
        # ========================================================

        return p_ndc, p_view, in_mask

    @torch.no_grad()
    def compute_cov_3d(self, s, r):
        L = build_scaling_rotation(s, r)
        cov3d = L @ L.transpose(1, 2)
        return cov3d

    @jaxtyped(typechecker=typechecked)
    @torch.no_grad()
    def compute_cov_2d(
        self,
        mean_3d: Float[torch.Tensor, "N 3"],
        cov_3d: Float[torch.Tensor, "N 3 3"],
        w2c: Float[torch.Tensor, "4 4"],
        f_x: Float[torch.Tensor, ""],
        f_y: Float[torch.Tensor, ""],
    ) -> Float[torch.Tensor, "N 2 2"]:
        """
        Projects 3D covariances to 2D image plane.

        Args:
        - mean_3d: Coordinates of center of 3D Gaussians.
        - cov_3d: 3D covariance matrix.
        - w2c: World-to-camera matrix.
        - f_x: Focal length along x-axis.
        - f_y: Focal length along y-axis.

        Returns:
        - cov_2d: 2D covariance matrix.
        """ 
        # ========================================================
        # TODO: Transform 3D mean coordinates to camera space
        # ========================================================

        # Transpose the rigid transformation part of the world-to-camera matrix
        mean_cam = (homogenize(mean_3d).unsqueeze(1) @ w2c).squeeze(1)    # (N,3)
        x, y, z = mean_cam[:,0], mean_cam[:,1], mean_cam[:,2]

        J = torch.zeros(mean_3d.shape[0], 3, 3).to(mean_3d)
        W = w2c[:3, :3].T

        J[:, 0, 0] = f_x / z
        J[:, 0, 2] = -f_x * x / (z ** 2)
        J[:, 1, 1] = f_y / z
        J[:, 1, 2] = -f_y * y / (z ** 2)
        # ========================================================
        # TODO: Compute Jacobian of view transform and projection
        cov_2d = J @ W @ cov_3d @ W.transpose(-1, -2) @ J.transpose(-1, -2)
        # ========================================================

        # add low pass filter here according to E.q. 32
        filter = torch.eye(2, 2).to(cov_2d) * 0.3
        return cov_2d[:, :2, :2] + filter[None]


    @jaxtyped(typechecker=typechecked)
    @torch.no_grad()
    def render(
        self,
        camera: Camera,
        mean_2d: Float[torch.Tensor, "N 2"],
        cov_2d: Float[torch.Tensor, "N 2 2"],
        color: Float[torch.Tensor, "N 3"],
        opacities: Float[torch.Tensor, "N 1"],
        depths: Float[torch.Tensor, "N"],
    ) -> Float[torch.Tensor, "H W 3"]:
        radii = get_radius(cov_2d)
        rect = get_rect(mean_2d, radii, width=camera.image_width, height=camera.image_height)

        pix_coord = torch.stack(
            torch.meshgrid(torch.arange(camera.image_height), torch.arange(camera.image_width), indexing='xy'),
            dim=-1,
        ).to(mean_2d.device)
        
        render_color = torch.ones(*pix_coord.shape[:2], 3).to(mean_2d.device)

        assert camera.image_height % self.tile_size == 0, "Image height must be divisible by the tile_size."
        assert camera.image_width % self.tile_size == 0, "Image width must be divisible by the tile_size."

        
        
        #print(render_color.shape)
        #print("width : ",camera.image_width)
        #print("height : ",camera.image_height)
        for h in range(0, camera.image_width, self.tile_size):
            for w in range(0, camera.image_height, self.tile_size):
                # check if the rectangle penetrate the tile
                over_tl = rect[0][..., 0].clip(min=w), rect[0][..., 1].clip(min=h)
                over_br = rect[1][..., 0].clip(max=w+self.tile_size-1), rect[1][..., 1].clip(max=h+self.tile_size-1)
                
                # a binary mask indicating projected Gaussians that lie in the current tile
                in_mask = (over_br[0] > over_tl[0]) & (over_br[1] > over_tl[1])
                if not in_mask.sum() > 0:
                    continue

                # ========================================================
                # TODO: Sort the projected Gaussians that lie in the current tile by their depths, in ascending order
                # ========================================================
                m = mean_2d[in_mask]            # [M, 2]
                cov = cov_2d[in_mask]           # [M, 2, 2]
                cols = color[in_mask]           # [M, 3]
                alphas = opacities[in_mask]  # [M]
                deps = depths[in_mask]          # [M]

                # 1. sort by depth ascending
                order = torch.argsort(deps, dim=0)
                m = m[order]
                cov = cov[order]
                cols = cols[order]
                alphas = alphas[order]
                # ========================================================
                # TODO: Compute the displacement vector from the 2D mean coordinates to the pixel coordinates
                # ========================================================
                tile_pix = pix_coord[h:h + self.tile_size, w:w + self.tile_size]  # [T, T, 2]
                # compute displacement to each Gaussian mean: [T, T, M, 2]
                disp = tile_pix.unsqueeze(0) - m.unsqueeze(1).unsqueeze(1)

                # ========================================================
                # TODO: Compute the Gaussian weight for each pixel in the tile
                # ========================================================
                inv_cov = torch.inverse(cov)
                quad = (torch.matmul(disp, inv_cov.unsqueeze(1))  * disp).sum(-1)
                weights = torch.exp(-0.5 * quad) 

                # ========================================================
                # TODO: Perform alpha blending
                # ========================================================
                opacity = weights[..., None] * alphas[..., None, None]
                accumulated_transparency = torch.cumprod(1.0 - opacity, dim=0)
                transmission = torch.cat(
                    [torch.ones_like(accumulated_transparency[:1]), accumulated_transparency[:-1]],
                    dim=0
                )
                weighted_colors = cols[:, None, None, :] * opacity * transmission
                tile_color = weighted_colors.sum(dim=0)
                residual_transparency = accumulated_transparency[-1].squeeze(-1).squeeze(-1)
                background_opacity = residual_transparency[..., None].expand(-1, -1, 3)
                tile_color += background_opacity
                #print("h = ",h + self.tile_size)
                #print("w = ",w + self.tile_size)
                i1,i2=h,h + self.tile_size
                j1,j2=w,w + self.tile_size
                render_color[
                    i1:i2,
                    j1:j2
                    
                ] = tile_color.reshape(self.tile_size, self.tile_size, -1)

        return render_color

@torch.no_grad()
def homogenize(points):
    """
    homogeneous points
    :param points: [..., 3]
    """
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)

@torch.no_grad()
def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

@torch.no_grad()
def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

@torch.no_grad()
def get_radius(cov2d):
    det = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] * cov2d[:, 1, 0]
    mid = 0.5 * (cov2d[:, 0, 0] + cov2d[:, 1, 1])
    lambda1 = mid + torch.sqrt((mid**2-det).clip(min=0.1))
    lambda2 = mid - torch.sqrt((mid**2-det).clip(min=0.1))
    return 3.0 * torch.sqrt(torch.max(lambda1, lambda2)).ceil()

@torch.no_grad()
def get_rect(pix_coord, radii, width, height):
    rect_min = (pix_coord - radii[:,None])
    rect_max = (pix_coord + radii[:,None])
    rect_min[..., 0] = rect_min[..., 0].clip(0, width - 1.0)
    rect_min[..., 1] = rect_min[..., 1].clip(0, height - 1.0)
    rect_max[..., 0] = rect_max[..., 0].clip(0, width - 1.0)
    rect_max[..., 1] = rect_max[..., 1].clip(0, height - 1.0)
    return rect_min, rect_max


### New function

def idx_manager(mean_3d,scales,rotations,shs,opacities,idx):
    if idx < 66 :
        return mirror(mean_3d,scales,rotations,shs,opacities)
    elif idx <95:
        return infiniteinception_back(mean_3d,scales,rotations,shs,opacities,1)
    elif idx <118:
        return infiniteinception_front(mean_3d,scales,rotations,shs,opacities,12)
    elif idx <201:
        return infiniteinception(mean_3d,scales,rotations,shs,opacities,12)
    else:
        return infiniteinception_front(mean_3d,scales,rotations,shs,opacities,12)

@jaxtyped(typechecker=typechecked)
@torch.no_grad()
def reflect_points(points):
    # this function creat a mirror in the plan x = ay + b
    a = 0.0
    b = 0.75
    # Normal vector of the plane: n = (0.1, -1, 0), normalized
    n = torch.tensor([1.0, a, 0.0], device=points.device, dtype=points.dtype)
    n = n / torch.norm(n)

    # A point on the plane: (0, -0.3, 0)
    p0 = torch.tensor([b, 0.0, 0.0], device=points.device, dtype=points.dtype)

    # Vector from p0 to each point
    vec = points - p0

    # Project vec onto normal, then reflect
    dot = torch.sum(vec * n, dim=1, keepdim=True)  # (N, 1)
    reflect = points - 2 * dot * n  # Apply reflection formula

    return reflect

@jaxtyped(typechecker=typechecked)
@torch.no_grad()
def alignz(mean_3d, scales, rotations, shs, opacities,z_deg):
    # ---------- pick the part we keep / mirror ------------------------------
    base_mask   = (mean_3d[:, 1] > -0.3)           # above mirror plane
    vis_mask    = (mean_3d[base_mask, 1] < 0.5)    # only show this slice

    pts  = mean_3d [base_mask][vis_mask]           # [N,3]
    scl  = scales   [base_mask][vis_mask]
    rot  = rotations[base_mask][vis_mask]          # quaternion (w,x,y,z)
    sh   = shs      [base_mask][vis_mask]
    opa  = opacities[base_mask][vis_mask]

    R_obj = transform(0, 0, z_deg, 0, 0, 0,
                      device=pts.device)[:3, :3]   # 3×3

    pts = (R_obj @ pts.T).T                       # rotate centres
    rot = quat_mul(matrix_to_quat(R_obj[None]),   # rotate quats
                   rot)
    
    return pts, scl, rot, sh, opa

def mirror(pts, scl, rot, sh, opa):
    # ---------- reflected centres ------------------------------------------
    pts_ref = reflect_points(pts)                  # y → −y (plane y=−0.3)

    # ---------- reflected orientation --------------------------------------
    M = torch.diag(torch.tensor([-1., 1., 1.], device=pts.device))  # reflection

    R      = build_rotation(rot)                   # [N,3,3]
    R_ref  = M @ R @ M                             # mirror the frame

    # SVD to pull out pure rotation (avoids scale pollution)
    # R_ref = U Σ Vᵀ  →  R_clean = U Vᵀ  (still det=+1)
    U, _, Vt = torch.linalg.svd(R_ref)
    R_clean  = U @ Vt

    rot_ref = matrix_to_quat(R_clean)              # back to quaternion

    # ---------- reflected SH lighting (flip y-odd bands) --------------------
    sh_ref = sh.clone()
    sh_ref[:, 1::2, :] *= -1                       # l = 1,3,5,…

    # ---------- stack base + mirror ----------------------------------------
    merged      = torch.cat([pts,  pts_ref], 0)
    scales_out  = torch.cat([scl,  scl     ], 0)
    rot_out     = torch.cat([rot,  rot_ref ], 0)
    sh_out      = torch.cat([sh,   sh_ref  ], 0)
    opa_out     = torch.cat([opa,  opa     ], 0)

    double = pts.clone()
    double[:,1] += 1.5

    merged = torch.cat([merged, double], dim=0)
    scales_out    = torch.cat([scales_out, scl], dim=0)
    rot_out = torch.cat([rot_out, rot], dim=0)
    sh_out       = torch.cat([sh_out, sh], dim=0)
    opa_out = torch.cat([opa_out, opa], dim=0)

    return merged, scales_out, rot_out, sh_out, opa_out


# helper: rotation-matrix → unit quaternion  (w,x,y,z)
def matrix_to_quat(R: torch.Tensor) -> torch.Tensor:
    m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    tr = m00 + m11 + m22
    qw = torch.sqrt(torch.clamp(tr + 1.0, min=1e-8)) * 0.5
    qx = torch.where(qw > 1e-4, (R[:, 2, 1] - R[:, 1, 2]) / (4*qw),
                     torch.sqrt(torch.clamp(1 + m00 - m11 - m22, min=1e-8)) * 0.5)
    qy = torch.where(qw > 1e-4,
                     (R[:, 0, 2] - R[:, 2, 0]) / (4*qw),
                     (R[:, 0, 1] + R[:, 1, 0]) / (4*qx + 1e-8))
    qz = torch.where(qw > 1e-4,
                     (R[:, 1, 0] - R[:, 0, 1]) / (4*qw),
                     (R[:, 0, 2] + R[:, 2, 0]) / (4*qx + 1e-8))
    quat = torch.stack([qw, qx, qy, qz], dim=1)
    return quat / quat.norm(dim=1, keepdim=True)






@jaxtyped(typechecker=typechecked)
@torch.no_grad()
def shift(mean_3d,scales,rotations,shs,opacities) :
    main = mean_3d.clone()
    reflect = mean_3d.clone()
    reflect[:, 1] += 0.2

    merged = torch.cat([main, reflect], dim=0)

    scales    = torch.cat([scales, scales], dim=0)
    rotations = torch.cat([rotations, rotations], dim=0)
    shs       = torch.cat([shs, shs], dim=0)
    opacities = torch.cat([opacities, opacities], dim=0)
    return merged,scales,rotations,shs,opacities


def filterfixe(mean_3d, scales, rotations, shs, opacities):

    double = mean_3d.clone()

    mask = (double[:, 0] < 0.15) & (double[:, 0] > -0.2) & \
            (double[:, 1] < 0.25) & (double[:, 1]  > -0.3) & \
            (double[:, 2] < 0.1) & (double[:, 2] > -0.362)
    
    mask = (double[:, 0] < 0.4) & (double[:, 0] > -0.4) & \
            (double[:, 1] < 0.4) & (double[:, 1]  > -0.4) & \
            (double[:, 2] < 0.1) & \
            (2* double[:, 2] + double[:, 1] > -0.9)  & (5 * double[:, 2] - double[:, 1] > -1.8) &\
            (double[:, 2] + double[:, 0] > -0.5) & (double[:, 2] - double[:, 0] > -0.4)
    

    double[:, 2] -= 0.2

    return (
        double[mask],
        scales[mask],
        rotations[mask],
        shs[mask],
        opacities[mask]
    )


@jaxtyped(typechecker=typechecked)
@torch.no_grad()
def filter(w2c,mean_3d,scales,rotations,shs,opacities) :
    T = w2c
    n = torch.tensor([0.0, 0.0, 1.0, 0.0], device = w2c.device)  # direction vector (w=0)

    # Point sur le plan : origine homogène
    p = torch.tensor([0.0, 0.0, 0.0, 1.0], device = w2c.device)  # position vector (w=1)
    T = T.to(w2c.device)

    # Transformation de la normale
    T_inv = torch.inverse(T)
    T_inv_T = T_inv.T
    n_transformed = T_inv_T @ n
    n_transformed = n_transformed[:3]

    # Transformation du point
    p_transformed = T @ p
    p_transformed = p_transformed[:3] / p_transformed[3]

    normal_unit = n_transformed / torch.norm(n_transformed)
    distance = -1.0
    p_transformed = p_transformed + distance * normal_unit

    vectors = mean_3d - p_transformed  # shape (N, 3)
    dot_products = torch.matmul(vectors, n_transformed)  # shape (N,)

    # Masques booléens
    mask_front = dot_products > 0     # points devant le plan
    mask_back = dot_products < 0.0
    return (
        mean_3d[mask_back],
        scales[mask_back],
        rotations[mask_back],
        shs[mask_back],
        opacities[mask_back]
    )


def duplication(mean_3d,scales,rotations,shs,opacities) :
    main = mean_3d.clone()
    reflect = mean_3d.clone()
    reflect[:, 1] += 0.7

    merged = torch.cat([main, reflect], dim=0)

    scales    = torch.cat([scales, scales], dim=0)
    rotations = torch.cat([rotations, rotations], dim=0)
    shs       = torch.cat([shs, shs], dim=0)
    opacities = torch.cat([opacities, opacities], dim=0)
    return merged,scales,rotations,shs,opacities


def infiniteinception(mean_3d,scales,rotations,shs,opacities,n) :
    d = 1.5 #space between each double
    alpha = 1.5 #number above 1, rate at wich the double quality decrease
    torch.manual_seed(42)

    main = mean_3d.clone()
    merged = mean_3d.clone()
    scales_merged = scales.clone()
    rotations_merged = rotations.clone()
    shs_merged = shs.clone()
    opacities_merged = opacities.clone()

    for i in range(n):

        mask = torch.rand(main.size(0)) < alpha**(-i)

        double_i = main[mask]
        scales_i = scales[mask]
        rotations_i= rotations[mask]
        shs_i = shs[mask]
        opacities_i = opacities[mask]

        double_i[:, 0] -= d*(i+1)
        double_i[:, 1] -= d*(i+1)
        for j in range(2*(i+1)+1):
            merged = torch.cat([merged, double_i], dim=0)
            scales_merged    = torch.cat([scales_merged, scales_i], dim=0)
            rotations_merged = torch.cat([rotations_merged, rotations_i], dim=0)
            shs_merged       = torch.cat([shs_merged, shs_i], dim=0)
            opacities_merged = torch.cat([opacities_merged, opacities_i], dim=0)
            double_i[:, 1] += d

    for i in range(n):

        mask = torch.rand(main.size(0)) < alpha**(-i)

        double_i = main[mask]
        scales_i = scales[mask]
        rotations_i= rotations[mask]
        shs_i = shs[mask]
        opacities_i = opacities[mask]

        double_i[:, 0] += d*(i+1)
        double_i[:, 1] -= d*(i+1)
        for j in range(2*(i+1)+1):
            merged = torch.cat([merged, double_i], dim=0)
            scales_merged    = torch.cat([scales_merged, scales_i], dim=0)
            rotations_merged = torch.cat([rotations_merged, rotations_i], dim=0)
            shs_merged       = torch.cat([shs_merged, shs_i], dim=0)
            opacities_merged = torch.cat([opacities_merged, opacities_i], dim=0)
            double_i[:, 1] += d

    for i in range(n):

        mask = torch.rand(main.size(0)) < alpha**(-i)

        double_i = main[mask]
        scales_i = scales[mask]
        rotations_i= rotations[mask]
        shs_i = shs[mask]
        opacities_i = opacities[mask]

        double_i[:, 1] -= d*(i+1)
        double_i[:, 0] -= d*(i)
        for j in range(2*(i)+1):
            merged = torch.cat([merged, double_i], dim=0)
            scales_merged    = torch.cat([scales_merged, scales_i], dim=0)
            rotations_merged = torch.cat([rotations_merged, rotations_i], dim=0)
            shs_merged       = torch.cat([shs_merged, shs_i], dim=0)
            opacities_merged = torch.cat([opacities_merged, opacities_i], dim=0)
            double_i[:, 0] += d

    for i in range(n):

        mask = torch.rand(main.size(0)) < alpha**(-i)

        double_i = main[mask]
        scales_i = scales[mask]
        rotations_i= rotations[mask]
        shs_i = shs[mask]
        opacities_i = opacities[mask]

        double_i[:, 1] += d*(i+1)
        double_i[:, 0] -= d*(i)
        for j in range(2*(i)+1):
            merged = torch.cat([merged, double_i], dim=0)
            scales_merged    = torch.cat([scales_merged, scales_i], dim=0)
            rotations_merged = torch.cat([rotations_merged, rotations_i], dim=0)
            shs_merged       = torch.cat([shs_merged, shs_i], dim=0)
            opacities_merged = torch.cat([opacities_merged, opacities_i], dim=0)
            double_i[:, 0] += d
        

    return merged,scales_merged,rotations_merged,shs_merged,opacities_merged


def infiniteinception_front(mean_3d,scales,rotations,shs,opacities,n) :
    d = 1.5 #space between each double
    alpha = 1.5 #number above 1, rate at wich the double quality decrease
    torch.manual_seed(42)

    main = mean_3d.clone()
    merged = mean_3d.clone()
    scales_merged = scales.clone()
    rotations_merged = rotations.clone()
    shs_merged = shs.clone()
    opacities_merged = opacities.clone()

    for i in range(n):

        mask = torch.rand(main.size(0)) < alpha**(-i)

        double_i = main[mask]
        scales_i = scales[mask]
        rotations_i= rotations[mask]
        shs_i = shs[mask]
        opacities_i = opacities[mask]

        double_i[:, 0] += d*(i+1)
        double_i[:, 1] -= d*(i+1)
        for j in range(2*(i+1)+1):
            merged = torch.cat([merged, double_i], dim=0)
            scales_merged    = torch.cat([scales_merged, scales_i], dim=0)
            rotations_merged = torch.cat([rotations_merged, rotations_i], dim=0)
            shs_merged       = torch.cat([shs_merged, shs_i], dim=0)
            opacities_merged = torch.cat([opacities_merged, opacities_i], dim=0)
            double_i[:, 1] += d

    for i in range(n):

        mask = torch.rand(main.size(0)) < alpha**(-i)

        double_i = main[mask]
        scales_i = scales[mask]
        rotations_i= rotations[mask]
        shs_i = shs[mask]
        opacities_i = opacities[mask]

        double_i[:, 1] += d*(i+1)
        double_i[:, 0] -= d*(i+1)
        for j in range(2*(i)+2):
            merged = torch.cat([merged, double_i], dim=0)
            scales_merged    = torch.cat([scales_merged, scales_i], dim=0)
            rotations_merged = torch.cat([rotations_merged, rotations_i], dim=0)
            shs_merged       = torch.cat([shs_merged, shs_i], dim=0)
            opacities_merged = torch.cat([opacities_merged, opacities_i], dim=0)
            double_i[:, 0] += d

    return merged,scales_merged,rotations_merged,shs_merged,opacities_merged


def infiniteinception_back(mean_3d,scales,rotations,shs,opacities,n) :
    d = 1.5 #space between each double
    alpha = 2.0 #number above 1, rate at wich the double quality decrease

    double_i = mean_3d.clone()
    merged = mean_3d.clone()
    scales_merged = scales.clone()
    rotations_merged = rotations.clone()
    shs_merged = shs.clone()
    opacities_merged = opacities.clone()

    double_i[:,0] -= d

    merged = torch.cat([merged, double_i], dim=0)
    scales_merged    = torch.cat([scales_merged, scales], dim=0)
    rotations_merged = torch.cat([rotations_merged, rotations], dim=0)
    shs_merged       = torch.cat([shs_merged, shs], dim=0)
    opacities_merged = torch.cat([opacities_merged, opacities], dim=0)

    double_i[:,0] += d
    double_i[:,1] -= d

    merged = torch.cat([merged, double_i], dim=0)
    scales_merged    = torch.cat([scales_merged, scales], dim=0)
    rotations_merged = torch.cat([rotations_merged, rotations], dim=0)
    shs_merged       = torch.cat([shs_merged, shs], dim=0)
    opacities_merged = torch.cat([opacities_merged, opacities], dim=0)

    double_i[:,1] += 2*d

    merged = torch.cat([merged, double_i], dim=0)
    scales_merged    = torch.cat([scales_merged, scales], dim=0)
    rotations_merged = torch.cat([rotations_merged, rotations], dim=0)
    shs_merged       = torch.cat([shs_merged, shs], dim=0)
    opacities_merged = torch.cat([opacities_merged, opacities], dim=0)

    return merged,scales_merged,rotations_merged,shs_merged,opacities_merged
    

def transform(x_deg, y_deg, z_deg, x_t, y_t, z_t, *, dtype=None, device=None):
    # angles → radians
    a, b, c = torch.deg2rad(torch.as_tensor([x_deg, y_deg, z_deg],
                                            dtype=dtype, device=device)) 
    ca, sa = torch.cos(a), torch.sin(a)
    cb, sb = torch.cos(b), torch.sin(b)
    cc, sc = torch.cos(c), torch.sin(c)

    R = torch.stack([
        torch.stack([cb*cc,  cc*sa*sb - ca*sc,  sa*sc + ca*cc*sb]),
        torch.stack([cb*sc,  ca*cc + sa*sb*sc,  ca*sb*sc - cc*sa]),
        torch.stack([-sb   ,  cb*sa           ,  ca*cb           ])
    ])                       

    T = torch.eye(4, dtype=dtype, device=device)
    T[:3, :3] = R
    T[:3,  3] = torch.as_tensor([x_t, y_t, z_t],
                                dtype=dtype, device=device)
    return T

def quat_mul(q1, q2):
    """
    Multiplication of two quaternions.
    """
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return torch.stack([w, x, y, z], dim=1)