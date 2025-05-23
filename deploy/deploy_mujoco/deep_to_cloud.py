import open3d as o3d
import numpy as np


class DeepToCloud:
    @classmethod
    def get_pcd(cls, rgb_img, depth_img):
        # Convert to Open3D images
        o3d_color = o3d.geometry.Image(rgb_img)
        o3d_depth = o3d.geometry.Image(depth_img)

        # Create RGBD image
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color,
            o3d_depth,
            depth_scale=1.0,
            depth_trunc=5.0,
            convert_rgb_to_intensity=False
        )

        # Set camera intrinsic parameters with a custom FOV
        width, height = depth_img.shape
        fov = 110  # Field of View in degrees
        fx = fy = (width / 2) / np.tan(np.radians(fov / 2))  # Focal length
        cx, cy = width / 2, height / 2  # Principal point

        intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

        # Create point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            intrinsic
        )
        
        rotation_matrix_x = pcd.get_rotation_matrix_from_xyz((-90 * np.pi / 180, 90 * np.pi / 180, 0))
        pcd.rotate(rotation_matrix_x)

        return pcd
