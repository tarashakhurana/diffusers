import torch
import struct
import numpy as np

from einops import repeat

img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))

def encode_plucker(ray_origins, ray_dirs, harmonic_embedding=None):
    """
    ray to plucker w/ pos encoding
    """
    plucker = torch.cat((ray_dirs, torch.cross(ray_origins, ray_dirs, dim=-1)), dim=-1)
    if harmonic_embedding is not None:
        plucker = harmonic_embedding(plucker)
    return plucker


def write_pointcloud(filename,xyz_points,rgb_points=None, edges=None):

    """ creates a .pkl file of the point clouds generated

    """

    assert xyz_points.shape[1] == 3,'Input XYZ points should be Nx3 float array'
    if rgb_points is None:
        rgb_points = np.ones(xyz_points.shape).astype(np.uint8)*255
    rgb_points = rgb_points.astype(np.uint8)
    assert xyz_points.shape == rgb_points.shape,'Input RGB colors should be Nx3 float array and have same size as input XYZ points'

    # Write header of .ply file
    fid = open(filename,'wb')
    fid.write(bytes('ply\n', 'utf-8'))
    fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
    fid.write(bytes('element vertex %d\n'%xyz_points.shape[0], 'utf-8'))
    fid.write(bytes('property float x\n', 'utf-8'))
    fid.write(bytes('property float y\n', 'utf-8'))
    fid.write(bytes('property float z\n', 'utf-8'))
    fid.write(bytes('property uchar red\n', 'utf-8'))
    fid.write(bytes('property uchar green\n', 'utf-8'))
    fid.write(bytes('property uchar blue\n', 'utf-8'))
    if edges is not None:
        fid.write(bytes('element edge %d\n'%edges.shape[0], 'utf-8'))
        fid.write(bytes('property int vertex1\n', 'utf-8'))
        fid.write(bytes('property int vertex2\n', 'utf-8'))
    fid.write(bytes('end_header\n', 'utf-8'))

    if edges is not None:
        print("shape of edges inside pt cloud function", edges.shape)

    # Write 3D points to .ply file
    for i in range(xyz_points.shape[0]):
        fid.write(bytearray(struct.pack("fffccc",xyz_points[i,0],xyz_points[i,1],xyz_points[i,2],
                                        rgb_points[i,0].tostring(),rgb_points[i,1].tostring(),
                                        rgb_points[i,2].tostring())))

    if edges is not None:
        for i in range(edges.shape[0]):
            fid.write(bytearray(struct.pack("ii",edges[i,0],edges[i,1])))

#@ FROM PyTorch3D
class HarmonicEmbedding(torch.nn.Module):
    def __init__(
        self,
        n_harmonic_functions: int = 6,
        omega_0: float = 1.0,
        logspace: bool = True,
        append_input: bool = True,
    ) -> None:
        """
        Given an input tensor `x` of shape [minibatch, ... , dim],
        the harmonic embedding layer converts each feature
        (i.e. vector along the last dimension) in `x`
        into a series of harmonic features `embedding`,
        where for each i in range(dim) the following are present
        in embedding[...]:
            ```
            [
                sin(f_1*x[..., i]),
                sin(f_2*x[..., i]),
                ...
                sin(f_N * x[..., i]),
                cos(f_1*x[..., i]),
                cos(f_2*x[..., i]),
                ...
                cos(f_N * x[..., i]),
                x[..., i],              # only present if append_input is True.
            ]
            ```
        where N corresponds to `n_harmonic_functions-1`, and f_i is a scalar
        denoting the i-th frequency of the harmonic embedding.
        If `logspace==True`, the frequencies `[f_1, ..., f_N]` are
        powers of 2:
            `f_1, ..., f_N = 2**torch.arange(n_harmonic_functions)`
        If `logspace==False`, frequencies are linearly spaced between
        `1.0` and `2**(n_harmonic_functions-1)`:
            `f_1, ..., f_N = torch.linspace(
                1.0, 2**(n_harmonic_functions-1), n_harmonic_functions
            )`
        Note that `x` is also premultiplied by the base frequency `omega_0`
        before evaluating the harmonic functions.
        Args:
            n_harmonic_functions: int, number of harmonic
                features
            omega_0: float, base frequency
            logspace: bool, Whether to space the frequencies in
                logspace or linear space
            append_input: bool, whether to concat the original
                input to the harmonic embedding. If true the
                output is of the form (x, embed.sin(), embed.cos()
        """
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                n_harmonic_functions,
                dtype=torch.float32,
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (n_harmonic_functions - 1),
                n_harmonic_functions,
                dtype=torch.float32,
            )

        self.register_buffer("_frequencies", frequencies * omega_0, persistent=False)
        self.append_input = append_input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor of shape [..., dim]
        Returns:
            embedding: a harmonic embedding of `x`
                of shape [..., (n_harmonic_functions * 2 + int(append_input)) * dim]
        """
        embed = (x[..., None] * self._frequencies).view(*x.shape[:-1], -1)
        embed = torch.cat(
            (embed.sin(), embed.cos(), x)
            if self.append_input
            else (embed.sin(), embed.cos()),
            dim=-1,
        )
        return embed

    @staticmethod
    def get_output_dim_static(
        input_dims: int,
        n_harmonic_functions: int,
        append_input: bool,
    ) -> int:
        """
        Utility to help predict the shape of the output of `forward`.
        Args:
            input_dims: length of the last dimension of the input tensor
            n_harmonic_functions: number of embedding frequencies
            append_input: whether or not to concat the original
                input to the harmonic embedding
        Returns:
            int: the length of the last dimension of the output tensor
        """
        return input_dims * (2 * n_harmonic_functions + int(append_input))

    def get_output_dim(self, input_dims: int = 3) -> int:
        """
        Same as above. The default for input_dims is 3 for 3D applications
        which use harmonic embedding for positional encoding,
        so the input might be xyz.
        """
        return self.get_output_dim_static(
            input_dims, len(self._frequencies), self.append_input
        )


def render_path_spiral(poses, focal, zrate=0.5, rots=2, N=120):
    """
        Construct a set of camera extrinsics that simulate the
        NeRF-like spiral path for forward facing scenes.
            poses: N x 3 x 5 array of poses
            focal: focal length for a reasonable "focus-depth" for the dataset
            zrate: rate of spiral along z-axis
            rots: number of rotations around spiral
            N: number of poses to sample
    """
    c2w = poses_avg(poses)
    up = normalize(poses[:, :3, 1].sum(0))
    tt = poses[:, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0)
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:, 4:5]

    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses


def poses_avg(poses):
    """
    Args:
        poses: N x 4 x 4
    """

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w


def normalize(x):
    """ normalize a vector """
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    """
        z: forward (1D vector with 3 entries)
        up: up (1D vector with 3 entries)
        pos: camera position (1D vector with 3 entries)
    """
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def get_points_given_imageplane(imageplane, depth, Rt, scale):
    Rt_inv = torch.linalg.inv(Rt)
    depth_in_cam = imageplane * depth * scale
    depth_in_cam = torch.cat([depth_in_cam, torch.ones_like(depth_in_cam)[..., 0:1, :, :]], dim=-3)

    if len(depth_in_cam.shape) == 3:
        depth_in_cam = depth_in_cam[None, None, ...]
    if len(depth_in_cam.shape) == 4:
        depth_in_cam = depth_in_cam[None, ...]

    if len(Rt_inv) == 2:
        Rt_inv = Rt_inv[None, None, ...]

    depth_in_world = torch.einsum("btnc,btchw->btnhw", Rt_inv, depth_in_cam)[..., :3, :, :]
    return depth_in_world


def get_plucker(K, Rt, H, W, return_image_plane=False, depth=None):
    Rt_inv = torch.linalg.inv(Rt)
    v, u = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    u = W - u - 1
    uv_homo = torch.stack([u.flatten(), v.flatten(), torch.ones_like(v.flatten())]).float()
    if depth is not None:
        cam_coords = torch.linalg.inv(K) @ uv_homo * depth
    else:
        cam_coords = torch.linalg.inv(K) @ uv_homo
    cam_coords_homo = torch.cat([cam_coords, torch.ones((1, cam_coords.shape[1]))], dim=0)

    ray_origins = repeat(Rt_inv[:3, 3], 'c -> c n', n=cam_coords_homo.shape[1]).T

    ray_dirs_unnormalized = Rt_inv @ cam_coords_homo
    ray_dirs_centered = ray_dirs_unnormalized[:3, :] - ray_origins.T
    ray_dirs = ray_dirs_centered.T / torch.norm(ray_dirs_centered.T, dim=-1, keepdim=True)

    if return_image_plane:
        return ray_origins, ray_dirs, cam_coords

    return ray_origins, ray_dirs


def topk_l1_error(pd, gt):
    l1_error = torch.mean(torch.abs(pd - gt), (1, 2, 3))
    return min(l1_error)


def topk_psnr(pd, gt):
    psnrs = [mse2psnr(img2mse(pd[i], gt[i])) for i in range(pd.shape[0])]
    return max(psnrs)


def topk_scaleshift_inv_l1_error(pd, gt):
    scale, shift = compute_scale_and_shift(pd, gt, torch.ones_like(pd))
    pd_ssi = scale[:, None, None, None] * pd + shift[:, None, None, None]
    return topk_l1_error(pd_ssi, gt)


def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2, 3))
    a_01 = torch.sum(mask * prediction, (1, 2, 3))
    a_11 = torch.sum(mask, (1, 2, 3))
    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2, 3))
    b_1 = torch.sum(mask * target, (1, 2, 3))
    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)
    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()
    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]
    return x_0, x_1


def make_gif(frames, save_path, duration):
    frame_one = frames[0]
    frame_one.save(save_path, format="GIF", append_images=frames, save_all=True, duration=duration, loop=0)


def draw_wireframe_camera(ax, world_T_camera=None, scale=1.0, color="b"):
    """
    Draw a wireframe camera in 3D matplotlib plot.
    Args:
        ax: matplotlib 3D axis
        world_T_camera: 4x4 transformation matrix from camera to world frame
    """
    # Camera body vertices
    vertices = np.array(
        [
            [-1, -1, 1.0],  # Front top-left
            [1, -1, 1.0],  # Front top-right
            [-1, 1, 1.0],  # Front bottom-left
            [1, 1, 1.0],  # Front bottom-right
            [0.0, 0.0, 0.0],  # Back (lens) center
        ]
    )
    vertices = vertices * scale

    # Apply the transformation matrix if provided
    if world_T_camera is not None:
        world_R_camera = world_T_camera[:3, :3]
        world_t_camera = world_T_camera[:3, 3]
        vertices = (world_R_camera @ vertices.T + world_t_camera[:, None]).T

    # Define edges by connecting vertices
    edges = [
        [vertices[0], vertices[1], vertices[3], vertices[2], vertices[0]],
        [vertices[0], vertices[2]],
        [vertices[1], vertices[3]],
        [vertices[0], vertices[4]],
        [vertices[1], vertices[4]],
        [vertices[2], vertices[4]],
        [vertices[3], vertices[4]],
    ]

    # Convert vertices and edges to numpy arrays
    vertices = np.array(vertices)
    edges = [np.array(edge) for edge in edges]

    # Plot the camera wireframe
    for edge in edges:
        ax.plot3D(*edge.T, color=color)

