import numpy as np
try:
    import cv2
except Exception:
    cv2 = None


class AnalysisService:
    def __init__(self, num_classes=3):
        self.num_classes = num_classes

    def compute_stats(self, mask):
        h, w = mask.shape
        total_area = h * w
        # 颜色语义：红色(2)为颗粒；轨迹为蓝色(1)+红色(2)
        particle_mask = (mask == 2).astype(np.uint8)
        track_mask = ((mask == 1) | (mask == 2)).astype(np.uint8)
        particle_area = int(particle_mask.sum())
        track_area = int(track_mask.sum())
        particle_ratio = particle_area / float(total_area) if total_area > 0 else 0.0
        # 颗粒数量按红色连通域统计
        particle_count = 0
        if cv2 is not None:
            num_labels, _ = cv2.connectedComponents(particle_mask)
            particle_count = max(0, num_labels - 1)
        else:
            particle_count = self._count_connected_components(particle_mask)
        return {
            'image_area': total_area,
            'particle_area': particle_area,
            'track_area': track_area,
            'particle_ratio': particle_ratio,
            'particle_count': particle_count,
        }

    def _count_connected_components(self, binary_mask):
        # 简易4邻域连通域统计（无opencv时的后备实现）
        visited = np.zeros_like(binary_mask, dtype=bool)
        count = 0
        h, w = binary_mask.shape
        for i in range(h):
            for j in range(w):
                if binary_mask[i, j] and not visited[i, j]:
                    count += 1
                    stack = [(i, j)]
                    visited[i, j] = True
                    while stack:
                        x, y = stack.pop()
                        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                            nx, ny = x+dx, y+dy
                            if 0 <= nx < h and 0 <= ny < w and binary_mask[nx, ny] and not visited[nx, ny]:
                                visited[nx, ny] = True
                                stack.append((nx, ny))
        return count

