# VLM-Ad-AIGC

**VLM-Based Automatic Product Advertisement Generation and Evaluation**

这是一个从商品图到广告关键帧、短视频并进行结构化评测的本地端到端示例项目。

## 示例结果

下面以一个商品示例记录完整的图像生成、视频生成和 VLM 评测结果。实验统一使用 **97 帧、24 FPS、576×768、约 4 秒、seed=20260909**。HunyuanVideo 当前生成结果存在明显噪声和色块异常，因此本示例只展示已保留的 SDXL/FLUX 与 LTXV/Wan2.2 组合。

## 表一：图像生成模型、视频生成模型与产物

VLM 不参与视频扩散生成，因此每个图像模型和视频模型组合只保留一个唯一 MP4。输入图片是商品原图，关键帧由对应图像模型生成，视频由对应视频模型根据关键帧生成。

| # | 图像生成模型 | 视频生成模型 | 输入图片 | 关键帧图 | 生成视频 |
|---:|---|---|---|---|---|
| 1 | SDXL 1.0 + IP-Adapter | LTXV-2B distilled | [输入图片](samples/example/input.png) | [SDXL 关键帧](samples/example/keyframes/sdxl_ip_adapter.png) | [查看视频](samples/example/videos/sdxl_ip_adapter__ltxv_2b.mp4) |
| 2 | SDXL 1.0 + IP-Adapter | Wan2.2 TI2V-5B | [输入图片](samples/example/input.png) | [SDXL 关键帧](samples/example/keyframes/sdxl_ip_adapter.png) | [查看视频](samples/example/videos/sdxl_ip_adapter__wan22_ti2v_5b.mp4) |
| 3 | FLUX.2-klein-4B | LTXV-2B distilled | [输入图片](samples/example/input.png) | [FLUX 关键帧](samples/example/keyframes/flux2_klein.png) | [查看视频](samples/example/videos/flux2_klein__ltxv_2b.mp4) |
| 4 | FLUX.2-klein-4B | Wan2.2 TI2V-5B | [输入图片](samples/example/input.png) | [FLUX 关键帧](samples/example/keyframes/flux2_klein.png) | [查看视频](samples/example/videos/flux2_klein__wan22_ti2v_5b.mp4) |

## 表二：VLM、生成模型与评测结果

VLM 仅负责从商品图提取商品类别、颜色、材质、包装结构和 OCR 等结构化信息，并参与评测记录；它不会改变图像生成或视频生成请求。因此下面 8 行是两种 VLM 的评测分支，但对应的 MP4 链接会指向同一个去重后的唯一视频。

| # | VLM | 图像生成模型 | 视频生成模型 | 生成视频 | 商品相似度 | 时序稳定性 | Storyboard |
|---:|---|---|---|---|---:|---:|---:|
| 1 | Qwen3-VL-4B-Instruct | SDXL 1.0 + IP-Adapter | LTXV-2B distilled | [查看视频](samples/example/videos/sdxl_ip_adapter__ltxv_2b.mp4) | 0.6843 | 0.6487 | 0.5530 |
| 2 | Qwen3-VL-4B-Instruct | SDXL 1.0 + IP-Adapter | Wan2.2 TI2V-5B | [查看视频](samples/example/videos/sdxl_ip_adapter__wan22_ti2v_5b.mp4) | 0.7809 | 0.9680 | 0.5581 |
| 3 | Qwen3-VL-4B-Instruct | FLUX.2-klein-4B | LTXV-2B distilled | [查看视频](samples/example/videos/flux2_klein__ltxv_2b.mp4) | 0.8852 | 0.8947 | 0.5936 |
| 4 | Qwen3-VL-4B-Instruct | FLUX.2-klein-4B | Wan2.2 TI2V-5B | [查看视频](samples/example/videos/flux2_klein__wan22_ti2v_5b.mp4) | 0.9862 | 0.8472 | 0.5244 |
| 5 | InternVL3.5-8B-HF | SDXL 1.0 + IP-Adapter | LTXV-2B distilled | [查看视频](samples/example/videos/sdxl_ip_adapter__ltxv_2b.mp4) | 0.6843 | 0.6487 | 0.5530 |
| 6 | InternVL3.5-8B-HF | SDXL 1.0 + IP-Adapter | Wan2.2 TI2V-5B | [查看视频](samples/example/videos/sdxl_ip_adapter__wan22_ti2v_5b.mp4) | 0.7809 | 0.9680 | 0.5581 |
| 7 | InternVL3.5-8B-HF | FLUX.2-klein-4B | LTXV-2B distilled | [查看视频](samples/example/videos/flux2_klein__ltxv_2b.mp4) | 0.8852 | 0.8947 | 0.5936 |
| 8 | InternVL3.5-8B-HF | FLUX.2-klein-4B | Wan2.2 TI2V-5B | [查看视频](samples/example/videos/flux2_klein__wan22_ti2v_5b.mp4) | 0.9862 | 0.8472 | 0.5244 |

指标说明：商品相似度是视频首帧、中间帧和末帧与商品原图的 masked DINO embedding 余弦相似度均值；时序稳定性和 Storyboard 是基于帧间变化与预设分镜的启发式分数，不等同于人工审美评分。

## 表三：使用的提示词（中英文）

图像模型共用图像提示词，视频模型共用视频提示词和负面提示词；VLM 不会改写这些文本。

| 类型 | 中文提示词 | English Prompt |
|---|---|---|
| 图像生成 Prompt `P-I` | 高级奢华美妆广告关键帧。仅以参考图作为商品身份来源。保持精确轮廓、容器数量、瓶盖或盖子几何、比例、配色、材质、表面质感及标签位置。将商品作为主角置于干净的编辑风格摄影棚场景：暖象牙色至香槟色渐变背景，洞石或磨砂玻璃台面，左上方柔和漫射主光，受控轮廓光，真实接触阴影，符合物理的反射，轻微景深，边缘清晰，高端化妆品广告摄影，8K 细节，竖构图，商品居中占画面 60%。包装保持连贯并可制造。不要添加其他商品或改变产品系列。若参考图中的印刷标记不清晰，则保留为抽象标记；只有在参考图中清晰可读时才保留，绝不臆造新的 Logo 或品牌名。参考商品类别：单个矩形玻璃香水瓶。保留参考图构图及产品系列。 | Premium luxury beauty advertising keyframe. Use the reference image as the only source of product identity. Preserve the exact silhouette, number of containers, cap or lid geometry, proportions, color palette, material, finish, and label placement. Place the product as the hero object in a clean editorial studio set: warm ivory-to-champagne gradient background, travertine or frosted glass surface, soft diffused key light from upper left, controlled rim light, realistic contact shadow, physically plausible reflections, subtle depth of field, crisp edges, high-end cosmetic campaign photography, 8k detail, vertical composition, centered product occupying 60 percent of the frame. Keep the packaging coherent and manufacturable. Do not add another product or alter the product family. Preserve any printed marks as abstract marks unless they are clearly legible in the reference; never hallucinate a new logo or brand name. Reference product class: single rectangular glass flacon. Preserve the reference composition and product family. |
| 视频生成 Prompt `P-V` | 根据输入关键帧制作一支精致的竖屏 4 秒奢华美妆广告。全片为一个连续镜头，不剪切。每一帧都保持准确的商品身份、轮廓、材质、颜色、瓶盖几何、标签位置和容器数量稳定。采用明确分镜：0.0–0.8 秒进行优雅的微距推进，展现表面纹理和高光；0.8–2.1 秒进行缓慢的 90° 顺时针环绕，产生平滑视差，同时商品保持固定；2.1–3.2 秒执行下方描述的克制型商品特写动作，并加入受控高光或液体闪光；3.2–4.04 秒保持平静的英雄镜头，并轻柔地稳定到最终构图。运动必须符合物理规律、具有电影感和高级感：光流平滑、地平线稳定、影棚光线柔和、阴影和反射真实、浅景深、无突然加速。商品必须始终是单一且连贯的物体，不得变形、复制、融化、拉伸或替换。不得出现手、人、文字叠加、水印或场景切换。商品专属动作：温暖的焦散光穿过琥珀色玻璃，同时金色瓶盖捕捉精确的轮廓高光。 | Create a polished vertical 4-second luxury beauty commercial from the supplied keyframe. One continuous shot, no cuts. Keep the exact product identity, silhouette, materials, colors, cap geometry, label placement, and number of containers stable in every frame. Use a clear storyboard: 0.0-0.8s an elegant macro push-in revealing surface texture and highlights; 0.8-2.1s a slow 90-degree clockwise camera orbit with smooth parallax while the product stays anchored; 2.1-3.2s a restrained product-specific micro-action described below, with a controlled specular highlight or liquid glint; 3.2-4.04s a calm hero hold and gentle settle on the final composition. Motion should be physically plausible, cinematic and premium: smooth optical-flow motion, stable horizon, soft studio lighting, realistic shadows and reflections, shallow depth of field, no abrupt acceleration. The product must remain a single coherent object; do not morph, duplicate, melt, stretch, or replace it. No hands, people, text overlays, watermarks, or scene cuts. Product-specific micro-action: a warm caustic light sweeps through the amber glass while the gold cap catches a precise rim highlight. |
| 负面 Prompt `P-N` | 包装扭曲、商品融化、商品重复、额外容器、轮廓变化、瓶盖变异、几何结构损坏、标签闪烁、虚构可读文字、Logo 幻觉、时间抖动、帧闪烁、相机抖动、滚动快门、硬切、跳切、突然缩放、物体漂浮、不可能的反射、液体变形、低分辨率、模糊、噪声、水印、字幕、边框。 | warped packaging, melted product, duplicated product, extra container, changing silhouette, cap mutation, broken geometry, flickering label, invented readable text, logo hallucination, temporal jitter, frame flicker, camera shake, rolling shutter, hard cut, jump cut, abrupt zoom, floating object, impossible reflection, deformed liquid, low resolution, blur, noise, watermark, subtitle, border |

## 运行链路

1. 准备本地权重目录，并通过 `AIGC_MODEL_ROOT` 指向它；权重不纳入 Git。
2. 启动 FastAPI GPU 网关：`bash scripts/start-phase2-service.sh`。
3. 运行示例矩阵：`python scripts/run-phase2-comparison.py --config configs/example-comparison.json`；`configs/phase2-comparison.json` 是同一可复现实例的完整 2×2×2 配置，可复制其中的 `samples` 条目加入自己的商品图。
4. 计算指标并生成报告：`python scripts/evaluate-phase2.py`，然后运行 `python scripts/report-phase2.py`。

默认矩阵为 **2 VLM × 2 图像模型 × 2 视频模型**。VLM 负责商品 JSON 理解与评测；SDXL + IP-Adapter 或 FLUX.2-klein 负责关键帧；LTXV 或 Wan2.2 负责 I2V；DINO/CLIP 和时序启发式脚本负责离线评测。服务端的 Python 依赖、API schema 和测试位于 `services/aigc-service/`。

## 目录

```text
configs/                 实验矩阵、提示词与输出规格
scripts/                 VLM、图像/视频生成、评测、报告和服务启动脚本
services/aigc-service/   FastAPI 网关、模型后端、schema 与测试
samples/example/         可直接查看的示例输入、关键帧和四个唯一视频
results/                 已完成实验的报告与 Phase 3 对照结果（本地生成）
```
