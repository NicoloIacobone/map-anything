import torch
from sam2_minimal.modeling.sam.mask_decoder import MaskDecoder
from sam2_minimal.modeling.sam.transformer import TwoWayTransformer
from sam2_minimal.modeling.sam.prompt_encoder import PromptEncoder

# Parametri di configurazione
embed_dim = 256
image_embedding_size = (64, 64)  # Dimensione dell'embedding dell'immagine (H, W)
input_image_size = (1024, 1024)  # Dimensione dell'immagine in input (H, W)
mask_in_chans = 16  # Numero di canali nascosti per l'encoding delle maschere

# Istanzia i componenti
# Instanziazione del PromptEncoder
prompt_encoder = PromptEncoder(
    embed_dim=embed_dim,
    image_embedding_size=image_embedding_size,
    input_image_size=input_image_size,
    mask_in_chans=mask_in_chans,
)

# Parametri di configurazione
depth = 2  # Numero di layer del transformer
num_heads = 8  # Numero di head per multihead attention
mlp_dim = 2048  # Dimensione interna del blocco MLP
attention_downsample_rate = 2  # Fattore di downsampling dell'attenzione

# Instanziazione del TwoWayTransformer
transformer = TwoWayTransformer(
    depth=depth,
    embedding_dim=embed_dim,
    num_heads=num_heads,
    mlp_dim=mlp_dim,
    attention_downsample_rate=attention_downsample_rate,
)
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