First terminal:
ssh -L 2006:localhost:2006 -L 2004:localhost:2004 eth
source ~/.bashrc
rerun --serve-web --port 2004 --web-viewer-port 2006

Second terminal:
ssh eth
source ~/.bashrc
python scripts/demo_images_only_inference.py --viz --image_folder