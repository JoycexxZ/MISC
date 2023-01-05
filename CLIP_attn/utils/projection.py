from email.mime import image
import torch
from torch.autograd import Function
import math


def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    if intrinsic_image_dim == image_dim:
        return intrinsic
    resize_width = int(math.floor(image_dim[1] * float(intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])))
    intrinsic[0][0] *= float(resize_width)/float(intrinsic_image_dim[0])
    intrinsic[1][1] *= float(image_dim[1])/float(intrinsic_image_dim[1])
    # account for cropping here
    intrinsic[0][2] *= float(image_dim[0]-1)/float(intrinsic_image_dim[0]-1)
    intrinsic[1][2] *= float(image_dim[1]-1)/float(intrinsic_image_dim[1]-1)
    return intrinsic

def euler2mat(angle):
    """Convert euler angles to rotation matrix.
     :param angle: [3] or [b, 3]
     :return
        rotmat: [3] or [b, 3, 3]
    source
    https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py
    """

    if len(angle.size()) == 1:
        x, y, z = angle[0], angle[1], angle[2]
        _dim = 0
        _view = [3, 3]
    elif len(angle.size()) == 2:
        b, _ = angle.size()
        x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]
        _dim = 1
        _view = [b, 3, 3]

    else:
        assert False

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    # zero = torch.zeros([b], requires_grad=False, device=angle.device)[0]
    # one = torch.ones([b], requires_grad=False, device=angle.device)[0]
    zero = z.detach()*0
    one = zero.detach()+1
    zmat = torch.stack([cosz, -sinz, zero,
                        sinz, cosz, zero,
                        zero, zero, one], dim=_dim).reshape(_view)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zero, siny,
                        zero, one, zero,
                        -siny, zero, cosy], dim=_dim).reshape(_view)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([one, zero, zero,
                        zero, cosx, -sinx,
                        zero, sinx, cosx], dim=_dim).reshape(_view)

    rot_mat = xmat @ ymat @ zmat
    # print(rot_mat)
    return rot_mat


class ProjectionHelper():
    def __init__(self, intrinsic, depth_min, depth_max, image_dims, f_accuracy, b_accuracy, cuda=True):
        self.intrinsic = intrinsic
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.image_dims = image_dims
        self.f_accuracy = f_accuracy
        self.b_accuracy = b_accuracy
        self.cuda = cuda

        # precompute
        self._compute_corner_points()

    def depth_to_skeleton(self, ux, uy, depth):
        # 2D to 3D coordinates with depth (used in compute_frustum_bounds)
        x = (ux - self.intrinsic[0][2]) / self.intrinsic[0][0]
        y = (uy - self.intrinsic[1][2]) / self.intrinsic[1][1]
        return torch.Tensor([depth*x, depth*y, depth])

    def skeleton_to_depth(self, p):
        x = (p[0] * self.intrinsic[0][0]) / p[2] + self.intrinsic[0][2]
        y = (p[1] * self.intrinsic[1][1]) / p[2] + self.intrinsic[1][2]
        return torch.Tensor([x, y, p[2]])
    
    def set_num_proposal(self, num):
        self.num_proposal = num
        
    def _compute_corner_points(self):
        if self.cuda:
            corner_points = torch.ones(8, 4).cuda()
        else:
            corner_points = torch.ones(8, 4)
        
        # image to camera
        # depth min
        corner_points[0][:3] = self.depth_to_skeleton(0, 0, self.depth_min)
        corner_points[1][:3] = self.depth_to_skeleton(self.image_dims[0] - 1, 0, self.depth_min)
        corner_points[2][:3] = self.depth_to_skeleton(self.image_dims[0] - 1, self.image_dims[1] - 1, self.depth_min)
        corner_points[3][:3] = self.depth_to_skeleton(0, self.image_dims[1] - 1, self.depth_min)
        # depth max
        corner_points[4][:3] = self.depth_to_skeleton(0, 0, self.depth_max)
        corner_points[5][:3] = self.depth_to_skeleton(self.image_dims[0] - 1, 0, self.depth_max)
        corner_points[6][:3] = self.depth_to_skeleton(self.image_dims[0] - 1, self.image_dims[1] - 1, self.depth_max)
        corner_points[7][:3] = self.depth_to_skeleton(0, self.image_dims[1] - 1, self.depth_max)

        self.corner_points = corner_points

    def compute_frustum_corners(self, camera_to_world):
        """
        Computes the coordinates of the viewing frustum corresponding to one image and given camera parameters

        :param camera_to_world: torch tensor of shape (4, 4)
        :return: corner_coords: torch tensor of shape (8, 4)
        """
        # input: camera pose (torch.Size([4, 4]))
        # output: coordinates of the corner points of the viewing frustum of the camera

        # corner_points = camera_to_world.new(8, 4, 1).fill_(1)

        # # image to camera
        # # depth min
        # corner_points[0][:3] = self.depth_to_skeleton(0, 0, self.depth_min).unsqueeze(1)
        # corner_points[1][:3] = self.depth_to_skeleton(self.image_dims[0] - 1, 0, self.depth_min).unsqueeze(1)
        # corner_points[2][:3] = self.depth_to_skeleton(self.image_dims[0] - 1, self.image_dims[1] - 1, self.depth_min).unsqueeze(1)
        # corner_points[3][:3] = self.depth_to_skeleton(0, self.image_dims[1] - 1, self.depth_min).unsqueeze(1)
        # # depth max
        # corner_points[4][:3] = self.depth_to_skeleton(0, 0, self.depth_max).unsqueeze(1)
        # corner_points[5][:3] = self.depth_to_skeleton(self.image_dims[0] - 1, 0, self.depth_max).unsqueeze(1)
        # corner_points[6][:3] = self.depth_to_skeleton(self.image_dims[0] - 1, self.image_dims[1] - 1, self.depth_max).unsqueeze(1)
        # corner_points[7][:3] = self.depth_to_skeleton(0, self.image_dims[1] - 1, self.depth_max).unsqueeze(1)


        # camera to world
        corner_coords = torch.bmm(camera_to_world.repeat(8, 1, 1), self.corner_points.unsqueeze(2))
        return corner_coords
    
    def compute_frustum_corners_batch(self, camera_to_world):
        """
        Computes the coordinates of the viewing frustum corresponding to one image and given camera parameters

        :param camera_to_world: torch tensor of shape (B, 4, 4)
        :return: corner_coords: torch tensor of shape (B, 8, 4)
        """
        corner_coords = torch.matmul(camera_to_world.unsqueeze(1), self.corner_points.unsqueeze(2)).squeeze(-1)
        return corner_coords
        
    def compute_frustum_normals(self, corner_coords):
        """
        Computes the normal vectors (pointing inwards) to the 6 planes that bound the viewing frustum

        :param corner_coords: torch tensor of shape (8, 4), coordinates of the corner points of the viewing frustum
        :return: normals: torch tensor of shape (6, 3)
        """

        normals = corner_coords.new(6, 3)

        # compute plane normals
        # front plane
        plane_vec1 = corner_coords[3][:3] - corner_coords[0][:3]
        plane_vec2 = corner_coords[1][:3] - corner_coords[0][:3]
        normals[0] = torch.cross(plane_vec1.view(-1), plane_vec2.view(-1))

        # right side plane
        plane_vec1 = corner_coords[2][:3] - corner_coords[1][:3]
        plane_vec2 = corner_coords[5][:3] - corner_coords[1][:3]
        normals[1] = torch.cross(plane_vec1.view(-1), plane_vec2.view(-1))

        # roof plane
        plane_vec1 = corner_coords[3][:3] - corner_coords[2][:3]
        plane_vec2 = corner_coords[6][:3] - corner_coords[2][:3]
        normals[2] = torch.cross(plane_vec1.view(-1), plane_vec2.view(-1))

        # left side plane
        plane_vec1 = corner_coords[0][:3] - corner_coords[3][:3]
        plane_vec2 = corner_coords[7][:3] - corner_coords[3][:3]
        normals[3] = torch.cross(plane_vec1.view(-1), plane_vec2.view(-1))

        # bottom plane
        plane_vec1 = corner_coords[1][:3] - corner_coords[0][:3]
        plane_vec2 = corner_coords[4][:3] - corner_coords[0][:3]
        normals[4] = torch.cross(plane_vec1.view(-1), plane_vec2.view(-1))

        # back plane
        plane_vec1 = corner_coords[6][:3] - corner_coords[5][:3]
        plane_vec2 = corner_coords[4][:3] - corner_coords[5][:3]
        normals[5] = torch.cross(plane_vec1.view(-1), plane_vec2.view(-1))

        return normals
    
    def compute_frustum_normals_batch(self, corner_coords):
        """
        Computes the normal vectors (pointing inwards) to the 6 planes that bound the viewing frustum

        :param corner_coords: torch tensor of shape (B, 8, 4), coordinates of the corner points of the viewing frustum
        :return: normals: torch tensor of shape (B, 6, 3)
        """
        B = corner_coords.shape[0]
        normals = corner_coords.new(B, 6, 3)

        # compute plane normals
        # front plane
        plane_vec1 = corner_coords[:, 3, :3] - corner_coords[:, 0, :3]
        plane_vec2 = corner_coords[:, 1, :3] - corner_coords[:, 0, :3]
        normals[:, 0] = torch.cross(plane_vec1, plane_vec2)

        # right side plane
        plane_vec1 = corner_coords[:, 2, :3] - corner_coords[:, 1, :3]
        plane_vec2 = corner_coords[:, 5, :3] - corner_coords[:, 1, :3]
        normals[:, 1] = torch.cross(plane_vec1, plane_vec2)

        # roof plane
        plane_vec1 = corner_coords[:, 3, :3] - corner_coords[:, 2, :3]
        plane_vec2 = corner_coords[:, 6, :3] - corner_coords[:, 2, :3]
        normals[:, 2] = torch.cross(plane_vec1, plane_vec2)

        # left side plane
        plane_vec1 = corner_coords[:, 0, :3] - corner_coords[:, 3, :3]
        plane_vec2 = corner_coords[:, 7, :3] - corner_coords[:, 3, :3]
        normals[:, 3] = torch.cross(plane_vec1, plane_vec2)

        # bottom plane
        plane_vec1 = corner_coords[:, 1, :3] - corner_coords[:, 0, :3]
        plane_vec2 = corner_coords[:, 4, :3] - corner_coords[:, 0, :3]
        normals[:, 4] = torch.cross(plane_vec1, plane_vec2)

        # back plane
        plane_vec1 = corner_coords[:, 6, :3] - corner_coords[:, 5, :3]
        plane_vec2 = corner_coords[:, 4, :3] - corner_coords[:, 5, :3]
        normals[:, 5] = torch.cross(plane_vec1, plane_vec2)

        return normals

    def points_in_frustum(self, corner_coords, normals, new_pts, return_mask=False):
        """
        Checks whether new_pts ly in the frustum defined by the coordinates of the corners coner_coords

        :param corner_coords: torch tensor of shape (8, 4), coordinates of the corners of the viewing frustum
        :param normals: torch tensor of shape (6, 3), normal vectors of the 6 planes of the viewing frustum
        :param new_pts: (num_points, 3)
        :param return_mask: if False, returns number of new_points in frustum
        :return: if return_mask=True, returns Boolean mask determining whether point is in frustum or not
        """

        # create vectors from point set to the planes
        point_to_plane1 = (new_pts.cuda() - corner_coords[2][:3].view(-1))
        point_to_plane2 = (new_pts.cuda() - corner_coords[4][:3].view(-1))

        # check if the scalar product with the normals is positive
        masks = list()
        # for each normal, create a mask for points that lie on the correct side of the plane
        for k, normal in enumerate(normals):
            if k < 3:
                masks.append(torch.round(torch.mm(point_to_plane1, normal.unsqueeze(1)) * 100) / 100 < 0)
            else:
                masks.append(torch.round(torch.mm(point_to_plane2, normal.unsqueeze(1)) * 100) / 100 < 0)
        mask = torch.ones(point_to_plane1.shape[0]) > 0
        mask = mask.cuda()

        # create a combined mask, which keeps only the points that lie on the correct side of each plane
        for addMask in masks:
            mask = mask * addMask.squeeze()

        if return_mask:
            return mask
        else:
            return torch.sum(mask)
        
    def points_in_frustum_batch(self, corner_coords, normals, new_pts, return_mask=False):
        """
        Checks whether new_pts ly in the frustum defined by the coordinates of the corners coner_coords

        :param corner_coords: torch tensor of shape (B, 8, 4), coordinates of the corners of the viewing frustum
        :param normals: torch tensor of shape (B, 6, 3), normal vectors of the 6 planes of the viewing frustum
        :param new_pts: (B, num_points, 3)
        :param return_mask: if False, returns number of new_points in frustum (B, )
        :return: if return_mask=True, returns Boolean mask determining whether point is in frustum or not (B, num_points)
        """
        # create vectors from point set to the planes
        point_to_plane1 = (new_pts.cuda() - corner_coords[:, 2, :3].unsqueeze(1)) # (B, num_points, 3)
        point_to_plane2 = (new_pts.cuda() - corner_coords[:, 4, :3].unsqueeze(1))

        # check if the scalar product with the normals is positive
        masks = list()
        # for each normal, create a mask for points that lie on the correct side of the plane
        for k in range(normals.shape[1]):
            if k < 3:
                masks.append(torch.round(torch.bmm(point_to_plane1, normals[:, k, :].unsqueeze(-1)) * 100) / 100 < 0)
            else:
                masks.append(torch.round(torch.bmm(point_to_plane2, normals[:, k, :].unsqueeze(-1)) * 100) / 100 < 0)
        mask = torch.ones((point_to_plane1.shape[0], point_to_plane1.shape[1])) > 0
        mask = mask.cuda()

        # create a combined mask, which keeps only the points that lie on the correct side of each plane
        for addMask in masks:
            mask = mask * addMask.squeeze()

        if return_mask:
            return mask
        else:
            return torch.sum(mask)
            
    def points_in_frustum_cpu(self, corner_coords, normals, new_pts, return_mask=False):
        """
        Checks whether new_pts ly in the frustum defined by the coordinates of the corners coner_coords

        :param corner_coords: torch tensor of shape (8, 4), coordinates of the corners of the viewing frustum
        :param normals: torch tensor of shape (6, 3), normal vectors of the 6 planes of the viewing frustum
        :param new_pts: (num_points, 3)
        :param return_mask: if False, returns number of new_points in frustum
        :return: if return_mask=True, returns Boolean mask determining whether point is in frustum or not
        """

        # create vectors from point set to the planes
        point_to_plane1 = (new_pts - corner_coords[2][:3].view(-1))
        point_to_plane2 = (new_pts - corner_coords[4][:3].view(-1))

        # check if the scalar product with the normals is positive
        masks = list()
        # for each normal, create a mask for points that lie on the correct side of the plane
        for k, normal in enumerate(normals):
            if k < 3:
                masks.append(torch.round(torch.mm(point_to_plane1, normal.unsqueeze(1)) * 100) / 100 < 0)
            else:
                masks.append(torch.round(torch.mm(point_to_plane2, normal.unsqueeze(1)) * 100) / 100 < 0)
        mask = torch.ones(point_to_plane1.shape[0]) > 0

        # create a combined mask, which keeps only the points that lie on the correct side of each plane
        for addMask in masks:
            mask = mask * addMask.squeeze()

        if return_mask:
            return mask
        else:
            return torch.sum(mask)
        
    def points_in_image_batch(self, points, align_matrices, poses, depths, image_mask):
        """
        Computes correspondances of points to pixels

        :param points: tensor containing all points of the point cloud (B, num_points, 3)
        :param align_matrices: align matrices (B, 4, 4)
        :param poses: camera pose (B, max_frame_num, 4, 4)
        :param depths: depth map (B, max_frame_num, 224, 224)
        :param image_mask: image mask (B, max_frame_num, 224, 224)
        :return: filter_inds (B, max_num_sel)
                 filter_points (B, max_num_sel, 3)
                 filter_num (B)
        """
        B, N, _ = points.shape
        F = poses.shape[1]
        if align_matrices is not None:
            inverse_align = torch.inverse(align_matrices.transpose(1, 2))
            _points = torch.ones((B, N, 4)).to(points.device)
            _points[:, :, :3] = points      
            _points = torch.bmm(_points, inverse_align)[:, :, :3]
        else:
            _points = points
        
        # compute viewing frustum
        # print(poses.shape)
        _poses = poses.reshape(B * F, 4, 4)
        corner_coords = self.compute_frustum_corners_batch(_poses)
        normals = self.compute_frustum_normals_batch(corner_coords)
        normals = normals.reshape(B, F, 6, 3)
        
        if depths is not None or image_mask is not None:
            coord = torch.ones((B, N, 4)).to(points.device)
            coord[:, :, :3] = _points
            coord = coord.unsqueeze(1).repeat(1, F, 1, 1).reshape(B*F, -1, 4)
            world_to_cam = torch.inverse(poses)
            world_to_cam = world_to_cam.reshape(B*F, 4, 4)
            coord = torch.bmm(world_to_cam, coord.transpose(1, 2)).transpose(1, 2) # (B*F, max_num_sel, 4)
            coord[:, :, 0] = (coord[:, :, 0] * self.intrinsic[0][0]) / coord[:, :, 2] + self.intrinsic[0][2]
            coord[:, :, 1] = (coord[:, :, 1] * self.intrinsic[1][1]) / coord[:, :, 2] + self.intrinsic[1][2]
            coord[:, :, :2] = torch.clamp(coord[:, :, :2], 0, self.image_dims[0]-1)
            image_ind = (torch.round(coord[:, :, 1]) * self.image_dims[0] + torch.round(coord[:, :, 0])).long() # (B*F, max_num_sel)
            
            coord = coord.reshape(B, F, -1, 4)
            image_ind = image_ind.reshape(B, F, -1)
        
        # check if points are in viewing frustum and only keep according indices
        point_mask = torch.ones((B, N)).cuda() < 0
        corner_coords = corner_coords.reshape(B, F, 8, 4)
        for f in range(F):
            mask = self.points_in_frustum_batch(corner_coords[:, f, :, :], normals[:, f, :, :], _points, return_mask=True).cuda() # (B, N)
            if depths is not None:
                image_ind_frame = image_ind[:, f, :] * mask
                depth = depths[:, f, :, :].reshape(B, -1)
                depth_val = torch.gather(depth, 1, image_ind_frame)
                depth_mask = ((depth_val - coord[:, f, :, 2]) < self.b_accuracy) & ((coord[:, f, :, 2] - depth_val) < self.f_accuracy)
                frame_mask = mask * depth_mask
            else:
                frame_mask = mask
            if image_mask is not None:
                image_ind_frame = image_ind[:, f, :]
                frame_image_mask = image_mask[:, f, :, :].reshape(B, -1) # (B, 224*224)
                img_mask = torch.gather(frame_image_mask, 1, image_ind_frame)
                frame_mask = frame_mask * img_mask
            point_mask = point_mask | frame_mask
        filter_num = point_mask.sum(dim=1)
        
        points_inds = torch.arange(0, N, out=torch.LongTensor()).unsqueeze(0).repeat(B, 1).cuda() # (B, N)
        if hasattr(self, 'num_proposal'):
            max_num_sel = self.num_proposal
        else:
            max_num_sel = filter_num.max()
            
        filter_points = torch.zeros((B, max_num_sel, 3)).to(points.device)
        filter_inds = -1 * torch.ones((B, max_num_sel)).to(points.device)
        for b in range(B):
            filter_points[b, :filter_num[b]] = points[b, point_mask[b]]
            filter_inds[b, :filter_num[b]] = points_inds[b, point_mask[b]]
            
        return filter_inds.long(), filter_points, filter_num

    def compute_projection(self, points, depth, image_mask, camera_to_world):
        """
        Computes correspondances of points to pixels

        :param points: tensor containing all points of the point cloud (num_points, 3)
        :param depth: depth map (size: proj_image)
        :param camera_to_world: camera pose (4, 4)
        :param num_points: number of points in one sample point cloud (4096)
        :return: indices_3d (array with point indices that correspond to a pixel),
                indices_2d (array with pixel indices that correspond to a point)
        """

        num_points = points.shape[0]
        world_to_camera = torch.inverse(camera_to_world)

        # create 1-dim array with all indices and array with 4-dim coordinates x, y, z, 1 of points
        ind_points = torch.arange(0, num_points, out=torch.LongTensor()).cuda()
        coords = camera_to_world.new(4, num_points)
        coords[:3, :] = torch.t(points)
        coords[3, :].fill_(1)

        # compute viewing frustum
        corner_coords = self.compute_frustum_corners(camera_to_world)
        normals = self.compute_frustum_normals(corner_coords)

        # check if points are in viewing frustum and only keep according indices
        mask_frustum_bounds = self.points_in_frustum(corner_coords, normals, points, return_mask=True).cuda()

        if not mask_frustum_bounds.any():
            print("No points in frustum")
            return None
        # ind_points = ind_points[mask_frustum_bounds]
        coords = coords[:, ind_points]
        print(coords.shape, world_to_camera.shape)
        # project world (coords) to camera
        camera = torch.mm(world_to_camera, coords)

        # project camera to image
        camera[0] = (camera[0] * self.intrinsic[0][0]) / camera[2] + self.intrinsic[0][2]
        camera[1] = (camera[1] * self.intrinsic[1][1]) / camera[2] + self.intrinsic[1][2]
        image = torch.round(camera).long()
        
        # print(camera.shape)

        # keep points that are projected onto the image into the correct pixel range
        valid_ind_mask = torch.ge(image[0], 0) * torch.ge(image[1], 0) * torch.lt(image[0], self.image_dims[0]) * torch.lt(image[1], self.image_dims[1])
        if not valid_ind_mask.any():
            return None
        valid_image_ind_x = image[0][valid_ind_mask]
        valid_image_ind_y = image[1][valid_ind_mask]
        valid_image_ind = valid_image_ind_y * self.image_dims[0] + valid_image_ind_x

        # keep only points that are in the correct depth ranges (self.depth_min - self.depth_max)
        if depth is not None:
            depth_vals = torch.index_select(depth.view(-1), 0, valid_image_ind.cuda())
            depth_mask = depth_vals.ge(self.depth_min) * depth_vals.le(self.depth_max) * \
                         (depth_vals - camera[2][valid_ind_mask]).le(self.b_accuracy) * \
                         (camera[2][valid_ind_mask] - depth_vals).le(self.f_accuracy)
            if not depth_mask.any():
                return None
            
        if image_mask is not None:
            img_mask = torch.index_select(image_mask.view(-1), 0, valid_image_ind.cuda())
            if not img_mask.any():
                return None

        # print(ind_points)
        # create two vectors for all considered points that establish 3d to 2d correspondence
        ind_update = ind_points[valid_ind_mask]
        camera = camera[:, valid_ind_mask]
        # print(ind_update)
        if depth is not None:
            ind_update = ind_update[depth_mask]
        if image_mask is not None:
            ind_update = ind_update[img_mask]
        # ind_update = ind_points[:]
        indices_3d = ind_update.new(num_points + 1).fill_(0) # needs to be same size for all in batch... (first element has size)
        indices_2d = ind_update.new(num_points + 1).fill_(0) # needs to be same size for all in batch... (first element has size)
        indices_3d[0] = ind_update.shape[0]  # first entry: number of relevant entries (of points)
        indices_2d[0] = ind_update.shape[0]
        indices_3d[1:1 + indices_3d[0]] = ind_update  # indices of points
        # if depth is not None:
        #     indices_2d[1:1 + indices_2d[0]] = torch.index_select(valid_image_ind, 0, torch.nonzero(depth_mask)[:, 0])  # indices of corresponding pixels
        # else:
        #     indices_2d[1:1 + indices_2d[0]] = valid_image_ind
        return indices_3d, None, camera

    @torch.no_grad()
    def project(self, label, lin_indices_3d, lin_indices_2d, num_points):
        """
        forward pass of backprojection for 2d features onto 3d points

        :param label: image features (shape: (num_input_channels, proj_image_dims[0], proj_image_dims[1]))
        :param lin_indices_3d: point indices from projection (shape: (num_input_channels, num_points_sample))
        :param lin_indices_2d: pixel indices from projection (shape: (num_input_channels, num_points_sample))
        :param num_points: number of points in one sample
        :return: array of points in sample with projected features (shape: (num_input_channels, num_points))
        """
        
        num_label_ft = 1 if len(label.shape) == 2 else label.shape[0] # = num_input_channels

        output = label.new(num_label_ft, num_points).fill_(0)
        num_ind = lin_indices_3d[0]
        if num_ind > 0:
            # selects values from image_features at indices given by lin_indices_2d
            vals = torch.index_select(label.view(num_label_ft, -1), 1, lin_indices_2d[1:1+num_ind])
            output.view(num_label_ft, -1)[:, lin_indices_3d[1:1+num_ind]] = vals
        
        return output


# Inherit from Function
class Projection(Function):

    @staticmethod
    def forward(ctx, label, lin_indices_3d, lin_indices_2d, num_points):
        """
        forward pass of backprojection for 2d features onto 3d points

        :param label: image features (shape: (num_input_channels, proj_image_dims[0], proj_image_dims[1]))
        :param lin_indices_3d: point indices from projection (shape: (num_input_channels, num_points_sample))
        :param lin_indices_2d: pixel indices from projection (shape: (num_input_channels, num_points_sample))
        :param num_points: number of points in one sample
        :return: array of points in sample with projected features (shape: (num_input_channels, num_points))
        """
        # ctx.save_for_backward(lin_indices_3d, lin_indices_2d)
        num_label_ft = 1 if len(label.shape) == 2 else label.shape[0] # = num_input_channels

        output = label.new(num_label_ft, num_points).fill_(0)
        num_ind = lin_indices_3d[0]
        if num_ind > 0:
            # selects values from image_features at indices given by lin_indices_2d
            vals = torch.index_select(label.view(num_label_ft, -1), 1, lin_indices_2d[1:1+num_ind])
            output.view(num_label_ft, -1)[:, lin_indices_3d[1:1+num_ind]] = vals
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_label = grad_output.clone()
        num_ft = grad_output.shape[0]
        grad_label.resize_(num_ft, 32, 41)
        lin_indices_3d, lin_indices_2d = ctx.saved_variables
        num_ind = lin_indices_3d.data[0]
        vals = torch.index_select(grad_output.data.contiguous().view(num_ft, -1), 1, lin_indices_3d.data[1:1+num_ind])
        grad_label.data.view(num_ft, -1)[:, lin_indices_2d.data[1:1+num_ind]] = vals
        
        return grad_label, None, None, None
