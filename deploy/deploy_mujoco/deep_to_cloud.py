import open3d as o3d


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

        # Create point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
            )
        )        
        return pcd
