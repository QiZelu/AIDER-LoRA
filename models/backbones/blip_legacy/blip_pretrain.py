"""Baseline BLIP encoder components, adapted from Salesforce BLIP."""
from torch import nn
from .med import BertConfig, BertModel
from .blip import create_vit, init_tokenizer


class BLIP_Pretrain(nn.Module):
    def __init__(self, med_config, bert_path, image_size=224, vit='large'):
        super().__init__()
        self.visual_encoder, vision_width = create_vit(vit, image_size, False, 0, 0)
        self.tokenizer = init_tokenizer(bert_path)
        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=encoder_config, add_pooling_layer=False)



