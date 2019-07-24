# Multi-targets multi-cameras tracking(MTMC)
本项目是一个用于公路全程监控系统跨摄像机多目标跟踪算法   

文件简介：  
- `Single_camera_track.py`单摄像机多目标跟踪
- `Multi_camera_track.py`跨摄像机多目标跟踪
- `data_generator.py` 数据生成器,用于同步输出多路图像
- `cameras_associate.py` 跨摄像机位置关联标定
- `Perspective_transform.py` 透视变换,主要用于图像坐标转世界坐标
- `Draw_trajectory.py` 轨迹绘制,在图像与画布平面上绘制目标轨迹
- `VideoMaker.py` 视频合成