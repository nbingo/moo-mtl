import torch

from models.pspnet.pspnet import get_segmentation_encoder, SegmentationDecoder


class PspNetWrapper(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.encoder = get_segmentation_encoder()
        self.segm_head = SegmentationDecoder(num_class=20, task_type='C')  # 19 + background
        self.inst_head = SegmentationDecoder(num_class=2, task_type='R')
        self.dept_head = SegmentationDecoder(num_class=1, task_type='R')
    

    def forward(self, data):
        x = data['data']
        x = self.encoder(x)

        return {
            'logits_segm': self.segm_head(x),
            'logits_inst': self.inst_head(x),
            'logits_depth': self.dept_head(x)
        }
    

    # this is required for cosmos
    def change_input_dim(self, dim):
        assert isinstance(dim, int)
        c = self.encoder.conv1
        self.encoder.conv1 = torch.nn.Conv2d(dim, c.out_channels, c.kernel_size, c.stride, c.padding, c.dilation, c.groups)