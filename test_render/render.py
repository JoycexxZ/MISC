from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    FoVOrthographicCameras, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    HardPhongShader,
    HardGouraudShader,
    SoftGouraudShader,
    TexturesVertex,
    TexturesAtlas,
    PointsRenderer,
    PointsRasterizationSettings,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor
)
from pytorch3d.renderer.mesh import Textures
from torch import nn
import numpy as np
import torch



class myRender():
    def __init__(self, image_size=224, points_per_pixel = 1, points_radius = 0.006, faces_per_pixel = 1, blur_radius = 0.0, background_color=(1.0, 1.0, 1.0)) -> None:
        
        self.image_size = image_size
        self.points_per_pixel = points_per_pixel
        self.points_radius = points_radius
        self.faces_per_pixel = faces_per_pixel
        self.blur_radius = blur_radius
        self.background_color = background_color
        
        
    def render_meshes(self, verts, faces, color=None):
        """_summary_

        Args:
            verts (Tensor): B * N * 3
            faces (Tensor): B * N2 * 3
        
        Return:
            images(Tensor): B * H * W range(0-1)
        """
        B, N, _ = verts.shape
        device = verts.device
        verts = [verts[i] for i in range(B)]
        faces = [faces[i] for i in range(B)]
        mesh = Meshes(
                    verts=verts,
                    faces=faces,
                    textures=None)
        mesh.textures = Textures(verts_rgb=torch.ones((B, N, 3), device=device))
        R, T = look_at_view_transform(eye=torch.Tensor([[0,0,3]]), elev = 0, azim = 0,
                                up=((-1, 0, 0),)) 
        cameras = FoVPerspectiveCameras(aspect_ratio=1,device=device, R=R, T=T)
        raster_settings = RasterizationSettings(
            image_size = self.image_size, 
            blur_radius = self.blur_radius, 
            faces_per_pixel = self.faces_per_pixel, 
            # max_faces_per_bin=1500
        )
        rasterizer = MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        )
        shader = HardPhongShader(device = device, cameras = cameras)
        renderer = MeshRenderer(rasterizer, shader)
        image = renderer(mesh)
        
        return image
    
    def render_points(self, points, color=None):
        """_summary_

        Args:
            points (Tensor): B * N * 3
            color (Tensor, optional): B * N * 3. Defaults to None.
        Return:
            images(Tensor): B * H * W range(0-1)
        """
        B, N, _ = points.shape
        device = points.device
        point_cloud = Pointclouds(points=points.to(torch.float), features=
                                  torch.zeros_like(points, dtype=torch.float), device=device)
        
        R, T = look_at_view_transform(dist = 3, elev = 
                                0, azim = 0) 
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        raster_settings = PointsRasterizationSettings(
            image_size=self.image_size,
            radius=self.points_radius,
            points_per_pixel=self.points_per_pixel
        )
        rasterizer = PointsRasterizer(
                    cameras=cameras,
                    raster_settings=raster_settings
        )
        compositor=NormWeightedCompositor( background_color=self.background_color )
        renderer = PointsRenderer(rasterizer, compositor)
        image = renderer(point_cloud)
        
        return image
        
        