import torch
from sam2_minimal.modeling.sam.mask_decoder import MaskDecoder
from sam2_minimal.modeling.sam.transformer import TwoWayTransformer
from sam2_minimal.modeling.sam.prompt_encoder import PromptEncoder

# Istanzia i componenti
prompt_encoder = PromptEncoder(embed_dim=256, image_size=64)  # controlla i parametri reali nel file
transformer = TwoWayTransformer(depth=2, embedding_dim=256)   # usa gli stessi default del repo
mask_decoder = MaskDecoder(transformer_dim=256, transformer=transformer, num_multimask_outputs=3)

# Input
B = 1
image_embeddings = torch.randn(B, 256, 64, 64)
image_pe = prompt_encoder.get_dense_pe()
sparse_emb = torch.zeros(B, 0, 256)       # nessun punto/box
dense_emb = torch.zeros(B, 256, 64, 64)   # oppure la dimensione prevista dal file (verifica nel repo)

# Forward
masks, iou_pred, sam_tokens, obj_scores = mask_decoder(
    image_embeddings=image_embeddings,
    image_pe=image_pe,
    sparse_prompt_embeddings=sparse_emb,
    dense_prompt_embeddings=dense_emb,
    multimask_output=True,
    repeat_image=False,
    high_res_features=None,
)