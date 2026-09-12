# Phase 3 消融报告

本报告复用已完成的 v0.1 视频，不重新生成媒体。所有数值都是商品一致性/时序/分镜的可解释代理指标，不能替代人工审片。

| 视频组合 | 真实 mask DINO mean | 中心椭圆 DINO mean | 真实-mask差值 | 16帧 DINO mean | 3帧 DINO mean | 事件级 Storyboard | 旧运动启发式 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `flux2_klein__ltxv_2b__perfume_flacon` | 0.72666 | 0.901 | -0.17434 | 0.72666 | 0.69604 | 0.855 | {'temporal_stability': 0.5157, 'motion_smoothness': 0.4189, 'dynamic_degree': 0.8636, 'note': 'retained for comparison; not the event-level Storyboard score'} |
| `flux2_klein__wan22_ti2v_5b__perfume_flacon` | 0.8996 | 0.98582 | -0.08622 | 0.8996 | 0.90255 | 0.565 | {'temporal_stability': 0.8142, 'motion_smoothness': 0.7771, 'dynamic_degree': 0.1602, 'note': 'retained for comparison; not the event-level Storyboard score'} |
| `sdxl_ip_adapter__ltxv_2b__perfume_flacon` | 0.48371 | 0.6719 | -0.18819 | 0.48371 | 0.52627 | 0.71 | {'temporal_stability': 0.1034, 'motion_smoothness': 0.0, 'dynamic_degree': 0.9816, 'note': 'retained for comparison; not the event-level Storyboard score'} |
| `sdxl_ip_adapter__wan22_ti2v_5b__perfume_flacon` | 0.5823 | 0.77828 | -0.19598 | 0.5823 | 0.62245 | 0.71 | {'temporal_stability': 0.8926, 'motion_smoothness': 0.8711, 'dynamic_degree': 0.1482, 'note': 'retained for comparison; not the event-level Storyboard score'} |

## 解释

- `真实 mask DINO` 使用输入商品与视频商品区域的 DINOv2 embedding 余弦相似度；中心椭圆仅作旧方法对照。
- `3帧/16帧` 对同一 MP4 使用不同的均匀采样密度；必须结合最低分、p10 和人工证据帧阅读。
- `事件级 Storyboard` 检查时间窗口、动作证据、顺序和商品是否缺失；`旧运动启发式` 只反映帧差与时长。
- 该单样本消融不支持跨数据集泛化或统计显著性结论。
