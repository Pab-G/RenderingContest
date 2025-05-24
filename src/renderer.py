"""
PyTorch implementation of Gaussian Splat Rasterizer.

The implementation is based on torch-splatting: https://github.com/hbb1/torch-splatting
"""

from jaxtyping import Bool, Float, jaxtyped
import torch

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
        self.tile_size = 36

    def render_scene(self, scene: Scene, camera: Camera):

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

        mean_3d, scales, rotations, shs, opacities = mirror(mean_3d, scales, rotations, shs, opacities)
        #mean_3d, scales, rotations, shs, opacities = shift(mean_3d, scales, rotations, shs, opacities)
        #mean_3d, scales, rotations, shs, opacities = mask(mean_3d, scales, rotations, shs, opacities)

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


@jaxtyped(typechecker=typechecked)
@torch.no_grad()
def reflect_points(points):
    # Creat a mirror in the plan y = ax + b
    a = 0.0
    b = -0.3

    # Normal vector of the plane: n = (0.1, -1, 0), normalized
    n = torch.tensor([a, -1.0, 0.0], device=points.device, dtype=points.dtype)
    n = n / torch.norm(n)

    # A point on the plane: (0, -0.3, 0)
    p0 = torch.tensor([0.0, b, 0.0], device=points.device, dtype=points.dtype)

    # Vector from p0 to each point
    vec = points - p0

    # Project vec onto normal, then reflect
    dot = torch.sum(vec * n, dim=1, keepdim=True)       # (N, 1)
    reflect = points - 2 * dot * n                      # Apply reflection formula

    return reflect


@jaxtyped(typechecker=typechecked)
@torch.no_grad()
def mirror(mean_3d, scales, rotations, shs, opacities):
    plane_y = -0.3

    # Original Reflection (already working correctly)
    mask = mean_3d[:, 1] > plane_y
    main_points = mean_3d[mask]
    reflected_points = reflect_points(main_points)

    scales_main = scales[mask]
    rotations_main = rotations[mask]
    shs_main = shs[mask] * 0.8  # Slight dimming for realism
    opacities_main = opacities[mask] * 0.8

    # Physical mirror plane: very thin and visually subtle
    mirror_size = 1.0  # Adjust mirror size
    num_points_side = 100
    lin = torch.linspace(-mirror_size, mirror_size, num_points_side).to(mean_3d.device)
    grid_x, grid_z = torch.meshgrid(lin, lin, indexing='ij')
    mirror_points = torch.stack([
        grid_x.reshape(-1),
        torch.full((num_points_side ** 2,), plane_y, device=mean_3d.device),
        grid_z.reshape(-1)
    ], dim=1)

    # Mirror surface appearance: subtle, flat color close to neutral white
    mirror_shs = torch.zeros((mirror_points.shape[0], shs.shape[1], 3), device=mean_3d.device)
    mirror_shs[:, 0, :] = 0.9  # neutral bright color
    mirror_scales = torch.full((mirror_points.shape[0], 3), 0.003, device=mean_3d.device)  # very small scales
    mirror_rotations = torch.tensor([[1, 0, 0, 0]] * mirror_points.shape[0], device=mean_3d.device).float()
    mirror_opacities = torch.full((mirror_points.shape[0], 1), 0.3, device=mean_3d.device)  # semi-transparent

    # Merge clearly
    merged_points = torch.cat([mean_3d, reflected_points, mirror_points], dim=0)
    merged_scales = torch.cat([scales, scales_main, mirror_scales], dim=0)
    merged_rotations = torch.cat([rotations, rotations_main, mirror_rotations], dim=0)
    merged_shs = torch.cat([shs, shs_main, mirror_shs], dim=0)
    merged_opacities = torch.cat([opacities, opacities_main, mirror_opacities], dim=0)

    return merged_points, merged_scales, merged_rotations, merged_shs, merged_opacities


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

def mask(mean_3d, scales, rotations, shs, opacities):
    double = mean_3d.clone()

    mask = ((double[:, 0] < -0.4) | (double[:, 0] > 0.1) | (double[:, 1] > -0.2) | (double[:, 2] > 0)) & (double[:, 1] < 0.5)

    return (
        double[mask],
        scales[mask],
        rotations[mask],
        shs[mask],
        opacities[mask]
    )