import numpy as np


class DeepToCloud:
    @classmethod
    def get_3d_point(cls, depth_img, width, height):
        intrinsics = np.array([
                                [600, 0, 320],  # fx, 0, cx
                                [0, 600, 240],  # 0, fy, cy
                                [0, 0, 1]       # 0, 0, 1
                            ])
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        # 创建像素坐标网格
        x = np.linspace(0, width - 1, width)
        y = np.linspace(0, height - 1, height)
        xv, yv = np.meshgrid(x, y)

        # 计算 3D 点
        z = depth_img
        x = (xv - cx) * z / fx
        y = (yv - cy) * z / fy

        # 将网格数据展平为点云
        points = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1)
        points = points[z.flatten() > 0]  # 去掉无效点

        return points