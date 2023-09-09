import torch
import struct
import numpy as np

from einops import repeat


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


def get_plucker(K, Rt, H, W, return_image_plane=False):
    Rt_inv = torch.linalg.inv(Rt)
    v, u = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    u = W - u - 1
    uv_homo = torch.stack([u, v, torch.ones_like(v)]).float()
    cam_coords = torch.linalg.inv(K) @ uv_homo.reshape(3, -1)
    cam_coords_homo = torch.cat([torch.ones((1, cam_coords.shape[1])), cam_coords], dim=0)

    ray_origins = repeat(Rt_inv[:3, 3], 'c -> c n', n=cam_coords_homo.shape[1]).T

    ray_dirs_unnormalized = Rt_inv @ cam_coords_homo
    ray_dirs_centered = ray_dirs_unnormalized[:3, :] - ray_origins.T
    ray_dirs = ray_dirs_centered.T / torch.norm(ray_dirs_centered.T, dim=-1, keepdim=True)

    if return_image_plane:
        return ray_origins, ray_dirs, ray_dirs_unnormalized[:3, :]

    return ray_origins, ray_dirs
