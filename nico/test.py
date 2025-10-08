import torch, json
from mapanything.models.mapanything.model import MapAnything

# Carica il JSON di esempio come base
cfg = json.load(open('/scratch2/nico/examples/meeting_11_09/class_attributes.json'))

# Istanzia dal blocco di init (puoi modificare encoders per evitare rete)
init_args = cfg['class_init_args']
# Esempio: prova a disattivare torch hub se serve offline (opzionale)
# init_args['encoder_config']['uses_torch_hub'] = False

model = MapAnything(**init_args)
model.eval()

# Mini input sintetico (1 view, B=1, H=W=128)
B, H, W = 1, 128, 128
img = torch.randn(B, 3, H, W)
views = [{
  'img': img,
  'data_norm_type': [model.encoder_config['data_norm_type']],  # tipicamente 'dinov2'
}]

with torch.inference_mode():
    preds = model.infer(
        views,
        memory_efficient_inference=True,
        use_amp=False  # più semplice per il test
    )

print('Infer OK. Num views:', len(preds))
print('Keys view0:', preds[0].keys())
# Se vuoi verificare che la tua head sia stata chiamata:
has_attr = hasattr(model, '_last_inst_embeddings')
print('instance embeddings present:', has_attr)
if has_attr:
    print('emb shape:', model._last_inst_embeddings.shape)