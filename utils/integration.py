import torch
import os
from os import path
from constants import color_palette
from PIL import Image
import cv2
import numpy as np

def visualize(masksem: torch.Tensor, size: int):
    mask, sem_map, other, obs = masksem[[0]].round(), masksem[1:-2], masksem[[-2]], masksem[[-1]]
    sem_map = torch.cat([
        mask * 1e-5, 
        obs * 1e-3,
        other * 1e-8,
        sem_map * 1e-0
    ], dim=0)
    sem_map = sem_map.argmax(0)
    sem_map += 2 * (sem_map > 2)
    sem_map = sem_map.cpu().numpy()

    color_pal = [int(x * 255.) for x in color_palette]
    sem_map_vis = Image.new("P", (sem_map.shape[1],
                                      sem_map.shape[0]))
    sem_map_vis.putpalette(color_pal)
    sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
    sem_map_vis = np.array(sem_map_vis.convert("RGB"))
    # sem_map_vis = sem_map_vis * mask + (sem_map_vis + 255) / 2 * (1 - mask) # 使未实际探索区域半透明
    sem_map_vis = np.flipud(sem_map_vis)

    sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]

    sem_map_vis = cv2.resize(sem_map_vis, (size, size),
                                interpolation=cv2.INTER_NEAREST)
    return sem_map_vis

class Collector:
    def __init__(self, scene_names: 'list[str]', path: str) -> None:
        if not isinstance(scene_names, list): scene_names = [scene_names]
        for i in range(len(scene_names)):
            scene_names[i] = self._preprocess_name(scene_names[i])
        self.names = scene_names
        self.path = path
        self._init_file_structure()
    def _init_file_structure(self):
        if not path.isdir(self.path):
            os.makedirs(self.path, exist_ok=True)
    def _init_file_structure_single(self, i):
        name = self.names[i]
        dir = path.join(self.path, name)
        if not path.isdir(dir):
            os.makedirs(dir, exist_ok=True)
    def _preprocess_name(self, name: str):
        return os.path.splitext(os.path.basename(name))[0]
    def update(self, i: int, new_name: str):
        self.names[i] = self._preprocess_name(new_name)

class EpisodeCollector(Collector):
    def __init__(self, scene_names: 'list[str]', frequency: float, threshold: int, path: str, started=True, preview_size=0) -> None:
        super().__init__(scene_names=scene_names, path=path)
        self.frequency = frequency if started else 0
        self.thereshold_abs = threshold
        self.started = started
        self.last_area_size = 0
        self.preview_size = preview_size
        self.update_cnt = [0] * len(self.names)
    def _init_file_structure(self):
        super()._init_file_structure()
        for i in range(len(self.names)):
            self._init_file_structure_single(i)
    def _preview(self):
        return self.preview_size > 0
    def update(self, i: int, new_name: str):
        res = super().update(i, new_name)
        self._init_file_structure_single(i)
        self.last_area_size[i] = 0
        self.update_cnt[i] += 1
        return res

    def collect(self, local_map: torch.Tensor):
        obstacle, mask, semantic = local_map[:, [0]].clamp(min=0, max=1).detach(), local_map[:, [1]].clamp(min=0, max=1).detach(), local_map[:, 4:].detach()
        masked_semantic = torch.cat([mask, semantic, obstacle], dim=1)
        area_size = mask.clamp(max=1).sum(list(range(len(mask.shape)))[1:])

        # manual_size = semantic.sum(dim=list(range(len(semantic.shape)))[1:])
        # assert (area_size - manual_size).abs().max() < 10, (area_size, manual_size)

        expansion = area_size - self.last_area_size
        largely_expanded = expansion >= self.thereshold_abs
        random_decision = torch.rand([mask.shape[0]]).to(largely_expanded.device) <= self.frequency
        collect = largely_expanded & random_decision
        for i, name in enumerate(self.names):
            if collect[i]:
                dir = path.join(self.path, name)
                cnt = len(os.listdir(dir))
                tensor_name = path.join(dir, str(cnt) + '-sparse' + '.masksem')
                torch.save(masked_semantic[i].to_sparse(), tensor_name)
                if self._preview():
                    vis = visualize(masked_semantic[i], self.preview_size)
                    img_name = path.join(dir, str(cnt) + '-' + str(i) + '-eps' + str(self.update_cnt[i]) + '.jpg')
                    cv2.imwrite(img_name, vis)
        self.last_area_size = (~collect) * self.last_area_size + collect * area_size
        
