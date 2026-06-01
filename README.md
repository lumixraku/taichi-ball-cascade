# Taichi Ball Cascade

2D 小球瀑布级联物理模拟，用 Taichi 实现。

## 运行

```bash
uv run python ball_physics.py
```

## 依赖

- Python 3.12+ / taichi 1.7.4 / numpy / uv

## Prompt 实现方式

用以下 prompt 可生成此项目：

```
用 taichi 做一个 2D 小球物理模拟：
- 球从顶部掉落，约每秒 5 个
- 7 层 30° 交错斜板，边缘在中间相接，形成瀑布级联
- 球沿斜板滑落，从缝隙掉到下一层，最终掉出画面自动删除
- 斜板用 mesh 渲染为木色填充矩形
- 纯 2D 物理（线段碰撞检测），无顶板底板
- 固定相机正对 XY 平面
```
