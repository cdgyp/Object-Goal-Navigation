import torch
import sys
sys.path.append('../../')
from models.perception.prediction.palette import prepare
    

class MapOutpainter(torch.nn.Module):
    def __init__(self, num_category: int, args, gpu_ids: 'list[int]', image_size=[240, 240]) -> None:
        super().__init__()
        self.model = prepare(
            class_number=num_category+1,
            base_config='models/perception/prediction/palette/config/simple_uncropping.json',
            batch_size=args.num_processes,
            gpu_ids=gpu_ids,
            epoch_per_train=1,
            iter_per_train=1000,
        )
        self.image_size = torch.Size(image_size)
        self.pool = torch.nn.AdaptiveAvgPool2d(image_size)
    def forward(self, full_map: torch.Tensor):
        """outpaint

        :param torch.Tensor full_map: 全局地图，形状为 B x (4 + num_category) x W x H，应该与 `main()` 中的 `full_map` 相同
        :return torch.Tensor: 预测的全局地图，形状为 B x num_category x W x H，应该可以直接 cat 在 `main()` 的 `full_map` 后面
        """
        assert full_map.requires_grad == False
        for i in [-1, -2]:  assert full_map.shape[i] % self.image_size[i] == 0

        obstacle, mask, semantic = full_map[:, [0]].clamp(0, 1), full_map[:, [1]].clamp(0, 1), full_map[:, 4:]
        semantic = torch.cat([semantic, obstacle], dim=1)

        mask, semantic = 1 - self.pool(mask).clamp(0, 1).round(), self.pool(semantic)
        inputs = torch.cat([semantic, mask], dim=1)
        outpainted: torch.Tensor = self.model(inputs)

        # 将 `outpainted` 放大到输入地图的大小
        outpainted = outpainted.unsqueeze(dim=-2).unsqueeze(dim=-1)
        new_shape = list(outpainted.shape)
        for i in [-1, -2]:
            r = int(full_map.shape[i] // self.image_size[i])
            new_shape[i * 2 + 1] *= r
        return outpainted.expand(new_shape).flatten(-2, -1).flatten(-3, -2)