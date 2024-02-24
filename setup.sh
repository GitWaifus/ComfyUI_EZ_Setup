#!/bin/bash

# Set up Python virtual environment
source /workspace/ComfyUI/venv/bin/activate

# Initial setup and cloning of repositories
pip install opencv-python
pip install simpleeval
pip install scikit-image==0.21.0
git config --global http.postBuffer 1048576000

# Clone repositories
git clone https://github.com/ltdrdata/ComfyUI-Manager.git /workspace/ComfyUI/custom_nodes/ComfyUI-Manager
git clone https://github.com/jags111/efficiency-nodes-comfyui.git /workspace/ComfyUI/custom_nodes/efficiency-nodes-comfyui 
git clone https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes.git /workspace/ComfyUI/custom_nodes/ComfyUI_Comfyroll_CustomNodes
git clone https://github.com/WASasquatch/was-node-suite-comfyui/ /workspace/ComfyUI/custom_nodes/was-node-suite-comfyui
git clone https://github.com/SeargeDP/SeargeSDXL.git /workspace/ComfyUI/custom_nodes/SeargeSDXL
git clone https://github.com/ssitu/ComfyUI_UltimateSDUpscale --recursive /workspace/ComfyUI/custom_nodes/ComfyUI_UltimateSDUpscale
git clone https://github.com/SLAPaper/ComfyUI-Image-Selector.git /workspace/ComfyUI/custom_nodes/ComfyUI-Image-Selector
git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git /workspace/ComfyUI/custom_nodes/ComfyUI-Impact-Pack
git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git /workspace/ComfyUI/custom_nodes/ComfyUI-Custom-Scripts
git clone https://github.com/FizzleDorf/ComfyUI_FizzNodes.git /workspace/ComfyUI/custom_nodes/ComfyUI_FizzNodes
git clone https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet.git /workspace/ComfyUI/custom_nodes/ComfyUI-Advanced-ControlNet
git clone https://github.com/Fannovel16/comfyui_controlnet_aux/ /workspace/ComfyUI/custom_nodes/comfyui_controlnet_aux
git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git /workspace/ComfyUI/custom_nodes/ComfyUI_IPAdapter_plus
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git /workspace/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite
git clone https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git /workspace/ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved
git clone https://github.com/twri/sdxl_prompt_styler.git /workspace/ComfyUI/custom_nodes/sdxl_prompt_styler.git 
git clone https://github.com/giriss/comfy-image-saver.git /workspace/ComfyUI/custom_nodes/comfy-image-saver
git clone https://github.com/aegis72/comfyui-styles-all.git /workspace/ComfyUI/custom_nodes/comfyui-styles-all
git clone https://github.com/yolain/ComfyUI-Easy-Use.git /workspace/ComfyUI/custom_nodes/ComfyUI-Easy-Use
git clone https://github.com/rgthree/rgthree-comfy.git /workspace/ComfyUI/custom_nodes/rgthree-comfy
git clone https://github.com/kijai/ComfyUI-DiffusersStableCascade.git /workspace/ComfyUI/custom_nodes/ComfyUI-DiffusersStableCascade
git clone https://github.com/cubiq/ComfyUI_essentials.git /workspace/ComfyUI/custom_nodes/ComfyUI_essentials 
git clone https://github.com/Extraltodeus/ComfyUI-AutomaticCFG.git /workspace/ComfyUI/custom_nodes/ComfyUI-AutomaticCFG
git clone https://github.com/crystian/ComfyUI-Crystools.git /workspace/ComfyUI/custom_nodes/ComfyUI-Crystools
git clone https://github.com/AIrjen/OneButtonPrompt /workspace/ComfyUI/custom_nodes/OneButtonPrompt


#wget -nc -O /workspace/ComfyUI/custom_nodes/ 'https://civitai.com/api/download/models/351034'

# Install requirements
pip install -r /workspace/ComfyUI/custom_nodes/was-node-suite-comfyui/requirements.txt
cd /workspace/ComfyUI/custom_nodes/efficiency-nodes-comfyui && pip install -r requirements.txt
cd /workspace/ComfyUI/custom_nodes/ComfyUI_FizzNodes && pip install -r requirements.txt
cd /workspace/ComfyUI/custom_nodes/ComfyUI-Advanced-ControlNet && pip install -r requirements.txt
cd /workspace/ComfyUI/custom_nodes/comfyui_controlnet_aux && pip install -r requirements.txt
cd /workspace/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite && pip install -r requirements.txt
cd /workspace/ComfyUI/custom_nodes/comfy-image-saver && pip install -r requirements.txt
cd /workspace/ComfyUI/custom_nodes/rgthree-comfy && pip install -r requirements.txt
cd /workspace/ComfyUI/custom_nodes/ComfyUI-DiffusersStableCascade && pip install -r requirements.txt
cd /workspace/ComfyUI/custom_nodes/ComfyUI_essentials && pip install -r requirements.txt
cd /workspace/ComfyUI/custom_nodes/ComfyUI-Crystools && pip install -r requirements.txt

# Download model files to checkpoints directory
mkdir -p /workspace/ComfyUI/models/checkpoints
wget -nc -O /workspace/ComfyUI/models/checkpoints/epicrealism_naturalSinRC1VAE.safetensors 'https://civitai.com/api/download/models/143906?type=Model&format=SafeTensor&size=pruned&fp=fp16'
wget -nc -O /workspace/ComfyUI/models/checkpoints/stable_cascade_stage_b.safetensors 'https://huggingface.co/stabilityai/stable-cascade/resolve/main/comfyui_checkpoints/stable_cascade_stage_b.safetensors?download=true'
wget -nc -O /workspace/ComfyUI/models/checkpoints/stable_cascade_stage_c.safetensors 'https://huggingface.co/stabilityai/stable-cascade/resolve/main/comfyui_checkpoints/stable_cascade_stage_c.safetensors?download=true'
wget -c https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors -P ./models/checkpoints/
wget -c https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors -P ./models/checkpoints/



# wget -nc -O /workspace/ComfyUI/models/checkpoints/ZavyComic.safetensors 'https://civitai.com/api/download/models/115754'
# wget -nc -O /workspace/ComfyUI/models/checkpoints/DreamShaperXLv21Turbo.safetensors 'https://civitai.com/api/download/models/351306'
# wget -nc -O /workspace/ComfyUI/models/checkpoints/aamXLAnimeMix_v10.safetensors 'https://civitai.com/api/download/models/303526?type=Model&format=SafeTensor&size=full&fp=fp16'
# wget -nc -O /workspace/ComfyUI/models/checkpoints/revAnimated_v122EOL.safetensors 'https://civitai.com/api/download/models/46846?type=Model&format=SafeTensor&size=full&fp=fp32'
# wget -nc -O /workspace/ComfyUI/models/checkpoints/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors 'https://huggingface.co/RunDiffusion/Juggernaut-XL-v9/resolve/main/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors?download=true'
# wget -nc -O /workspace/ComfyUI/models/checkpoints/CyberRealistic_V4.1_FP32.safetensors 'https://huggingface.co/cyberdelia/CyberRealistic/resolve/main/CyberRealistic_V4.1_FP32.safetensors?download=true'
# wget -nc -O /workspace/ComfyUI/models/checkpoints/analogMadness_v70.safetensors 'https://civitai.com/api/download/models/261539'
# wget -nc -O /workspace/ComfyUI/models/checkpoints/dreamshaper_8.safetensors 'https://civitai.com/api/download/models/128713?type=Model&format=SafeTensor&size=pruned&fp=fp16'
# wget -nc -O /workspace/ComfyUI/models/checkpoints/realisticVisionV60B1_v51VAE.safetensors 'https://civitai.com/api/download/models/130072?type=Model&format=SafeTensor&size=full&fp=fp16'
# wget -nc -O /workspace/ComfyUI/models/checkpoints/abyssorangemix3AOM3_aom3a1b.safetensors 'https://civitai.com/api/download/models/17233'
# wget -nc -O /workspace/ComfyUI/models/checkpoints/epicphotogasm_lastUnicorn.safetensors 'https://civitai.com/api/download/models/223670?type=Model&format=SafeTensor&size=pruned&fp=fp16'
# wget -nc -O /workspace/ComfyUI/models/checkpoints/
# wget -nc -O /workspace/ComfyUI/models/checkpoints/
# wget -nc -O /workspace/ComfyUI/models/checkpoints/

# Download model files to UNET folder
# wget -nc -O /workspace/ComfyUI/models/unet/stage_b_bf16.safetensors 'https://huggingface.co/stabilityai/stable-cascade/resolve/main/stage_b_bf16.safetensors?download=true'
# wget -nc -O /workspace/ComfyUI/models/unet/stage_c_bf16.safetensors 'https://huggingface.co/stabilityai/stable-cascade/resolve/main/stage_c_bf16.safetensors?download=true'

# Download model files to loras directory
mkdir -p /workspace/ComfyUI/models/loras
wget -nc -O /workspace/ComfyUI/models/loras/sd_xl_offset_example-lora_1.0.safetensors 'https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_offset_example-lora_1.0.safetensors'
wget -nc -O /workspace/ComfyUI/models/loras/aesthetic_anime_v1s.safetensors 'https://civitai.com/api/download/models/331598?type=Model&format=SafeTensor'
wget -nc -O /workspace/ComfyUI/models/loras/more_details.safetensors 'https://civitai.com/api/download/models/87153?type=Model&format=SafeTensor'
wget -nc -O /workspace/ComfyUI/models/loras/CharacterDesign_Concept-10.safetensors 'https://civitai.com/api/download/models/107502?type=Model&format=SafeTensor'
wget -nc -O /workspace/ComfyUI/models/loras/DetailedEyes_XLv3.safetensors 'https://civitai.com/api/download/models/145907'
wget -nc -O /workspace/ComfyUI/models/loras/animetarotV51.safetensors 'https://civitai.com/api/download/models/28609'
wget -nc -O /workspace/ComfyUI/models/loras/pixel_f2.safetensors 'https://civitai.com/api/download/models/52870?type=Model&format=SafeTensor'
wget -nc -O /workspace/ComfyUI/models/loras/ghibli_style_offset.safetensors  'https://civitai.com/api/download/models/7657?type=Model&format=SafeTensor&size=full&fp=fp16'
wget -nc -O /workspace/ComfyUI/models/loras/elf-pc-98.safetensors 'https://civitai.com/api/download/models/14769?type=Model&format=SafeTensor&size=full&fp=fp16'
wget -nc -O /workspace/ComfyUI/models/loras/GachaSplash4.safetensors 'https://civitai.com/api/download/models/38884'
wget -nc -O /workspace/ComfyUI/models/loras/epiCRealismHelper.safetensors 'https://civitai.com/api/download/models/118945?type=Model&format=SafeTensor'
wget -nc -O /workspace/ComfyUI/models/loras/lucy_offset.safetensors 'https://civitai.com/api/download/models/6370?type=Model&format=SafeTensor'
wget -nc -O /workspace/ComfyUI/models/loras/polyhedron_new_skin_v1.1.safetensors 'https://civitai.com/api/download/models/122580'
wget -nc -O /workspace/ComfyUI/models/loras/arcane_offset.safetensors 'https://civitai.com/api/download/models/8339?type=Model&format=SafeTensor&size=full&fp=fp16'
wget -nc -O /workspace/ComfyUI/models/loras/animemix_v3_offset.safetensors 'https://civitai.com/api/download/models/60568?type=Model&format=SafeTensor'
wget -nc -O /workspace/ComfyUI/models/loras/juliaAP1.0.safetensors 'https://civitai.com/api/download/models/226146?type=Model&format=SafeTensor'
wget -nc -O /workspace/ComfyUI/models/loras/thickline_fp16.safetensors 'https://civitai.com/api/download/models/16368'
wget -nc -O /workspace/ComfyUI/models/loras/Constricted PupilsV3.safetensors 'https://civitai.com/api/download/models/91871'
wget -nc -O /workspace/ComfyUI/models/loras/genshinfull1-000006.safetensors 'https://civitai.com/api/download/models/116970?type=Model&format=SafeTensor'|
wget -c https://civitai.com/api/download/models/10350 -O ./models/loras/theovercomer8sContrastFix_sd21768.safetensors #theovercomer8sContrastFix SD2.x 768-v
wget -c https://civitai.com/api/download/models/10638 -O ./models/loras/theovercomer8sContrastFix_sd15.safetensors #theovercomer8sContrastFix SD1.x

#wget -nc -O /worksapce/ComfyUI/models/loras/Harrlogos_v2.0.safetensors 'https://civitai.com/api/download/models/214296' // this doesnt work idk why 

# Download upscale model files
mkdir -p /workspace/ComfyUI/models/upscale_models
wget -nc -O /workspace/ComfyUI/models/upscale_models/4x-UltraSharp.pth 'https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x-UltraSharp.pth'
wget -nc -O /workspace/ComfyUI/models/upscale_models/4x_NMKD-Siax_200k.pth 'https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x_NMKD-Siax_200k.pth'
wget -nc -O /workspace/ComfyUI/models/upscale_models/4x_Nickelback_70000G.pth 'https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x_Nickelback_70000G.pth'
wget -nc -O /workspace/ComfyUI/models/upscale_models/1x-ITF-SkinDiffDetail-Lite-v1.pth 'https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/1x-ITF-SkinDiffDetail-Lite-v1.pth'

# ESRGAN upscale model
wget -nc https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P ./models/upscale_models/
wget -nc https://huggingface.co/sberbank-ai/Real-ESRGAN/resolve/main/RealESRGAN_x2.pth -P ./models/upscale_models/
wget -nc https://huggingface.co/sberbank-ai/Real-ESRGAN/resolve/main/RealESRGAN_x4.pth -P ./models/upscale_models/

# Create and download annotator model files
mkdir -p /workspace/ComfyUI/models/annotators
wget -nc -O /workspace/ComfyUI/models/annotators/ControlNetHED.pth 'https://huggingface.co/lllyasviel/Annotators/resolve/main/ControlNetHED.pth'
wget -nc -O /workspace/ComfyUI/models/annotators/res101.pth 'https://huggingface.co/lllyasviel/Annotators/resolve/main/res101.pth'

# Download clip_vision model file
mkdir -p /workspace/ComfyUI/models/clip_vision
wget -nc -O /workspace/ComfyUI/models/clip_vision/clip_vision_g.safetensors 'https://huggingface.co/stabilityai/control-lora/resolve/main/revision/clip_vision_g.safetensors'

# Download clip model file
mkdir -p /workspace/ComfyUI/models/clip
wget -nc -O /workspace/ComfyUI/models/clip/model.safetensors 'https://huggingface.co/stabilityai/stable-cascade/resolve/main/text_encoder/model.safetensors'

# ControlNet
wget -c https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11e_sd15_ip2p_fp16.safetensors -P ./models/controlnet/
wget -c https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11e_sd15_shuffle_fp16.safetensors -P ./models/controlnet/
wget -c https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_canny_fp16.safetensors -P ./models/controlnet/
wget -c https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11f1p_sd15_depth_fp16.safetensors -P ./models/controlnet/
wget -c https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_inpaint_fp16.safetensors -P ./models/controlnet/
wget -c https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_lineart_fp16.safetensors -P ./models/controlnet/
wget -c https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_mlsd_fp16.safetensors -P ./models/controlnet/
wget -c https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_normalbae_fp16.safetensors -P ./models/controlnet/
wget -c https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_openpose_fp16.safetensors -P ./models/controlnet/
wget -c https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_scribble_fp16.safetensors -P ./models/controlnet/
wget -c https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_seg_fp16.safetensors -P ./models/controlnet/
wget -c https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_softedge_fp16.safetensors -P ./models/controlnet/
wget -c https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15s2_lineart_anime_fp16.safetensors -P ./models/controlnet/
wget -c https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11u_sd15_tile_fp16.safetensors -P ./models/controlnet/

# ControlNet SDXL
wget -c https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/control-lora-canny-rank256.safetensors -P ./models/controlnet/
wget -c https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/control-lora-depth-rank256.safetensors -P ./models/controlnet/
wget -c https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/control-lora-recolor-rank256.safetensors -P ./models/controlnet/
wget -c https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/control-lora-sketch-rank256.safetensors -P ./models/controlnet/

# Controlnet Preprocessor nodes by Fannovel16
cd /workspace/ComfyUI/custom_nodes && git clone https://github.com/Fannovel16/comfy_controlnet_preprocessors; cd comfy_controlnet_preprocessors && python install.py


# GLIGEN
wget -c https://huggingface.co/comfyanonymous/GLIGEN_pruned_safetensors/resolve/main/gligen_sd14_textbox_pruned_fp16.safetensors -P ./models/gligen/


# Download control-lora model files
mkdir -p /workspace/ComfyUI/models/controlnet
wget -nc -O /workspace/ComfyUI/models/controlnet/control-lora-canny-rank256.safetensors 'https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/control-lora-canny-rank256.safetensors'
wget -nc -O /workspace/ComfyUI/models/controlnet/control-lora-depth-rank256.safetensors 'https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/control-lora-depth-rank256.safetensors'
wget -nc -O /workspace/ComfyUI/models/controlnet/control-lora-recolor-rank256.safetensors 'https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/control-lora-recolor-rank256.safetensors'
wget -nc -O /workspace/ComfyUI/models/controlnet/control-lora-sketch-rank256.safetensors 'https://huggingface.co/stabilityai/control-lora/resolve/main/control-LoRAs-rank256/control-lora-sketch-rank256.safetensors'

# Download embeddings model files
mkdir -p /workspace/ComfyUI/models/embeddings
wget -nc -O /workspace/ComfyUI/models/embeddings/bad_prompt_version2-neg.pt 'https://civitai.com/api/download/models/60095?type=Negative&format=Other'
wget -nc -O /workspace/ComfyUI/models/embeddings/badhandv4.pt 'https://civitai.com/api/download/models/20068'
wget -nc -O /workspace/ComfyUI/models/embeddings/badartist.pt 'https://civitai.com/api/download/models/6056?type=Negative&format=Other'
wget -nc -O /workspace/ComfyUI/models/embeddings/negativeXL.safetensors 'https://civitai.com/api/download/models/134583'
wget -nc -O /workspace/ComfyUI/models/embeddings/BadDream.pt 'https://civitai.com/api/download/models/77169?type=Model&format=PickleTensor'
wget -nc -O /workspace/ComfyUI/models/embeddings/ac_neg1.safetensors 'https://civitai.com/api/download/models/166373?type=Model&format=SafeTensor'
wget -nc -O /workspace/ComfyUI/models/embeddings/ng_deepnegative_v1_75t.pt 'https://civitai.com/api/download/models/5637?type=Model&format=PickleTensor&size=full&fp=fp16'
wget -nc -O /workspace/ComfyUI/models/embeddings/easynegative.safetensors 'https://civitai.com/api/download/models/9208?type=Model&format=SafeTensor&size=full&fp=fp16'
wget -nc -O /workspace/ComfyUI/models/embeddings/AS-YoungV2-neg.pt 'https://civitai.com/api/download/models/94765'
wget -nc -O /workspace/ComfyUI/models/embeddings/negative_hand.pt 'https://civitai.com/api/download/models/60938?type=Negative&format=Other'
wget -nc -O /workspace/ComfyUI/models/embeddings/FastNegativeV2.pt 'https://civitai.com/api/download/models/94057?type=Model&format=PickleTensor'
wget -nc -O /workspace/ComfyUI/models/embeddings/verybadimagenegative_v1.3.pt 'https://civitai.com/api/download/models/25820?type=Model&format=PickleTensor&size=full&fp=fp16'
wget -nc -O /workspace/ComfyUI/models/embeddings/epiCNegative.pt 'https://civitai.com/api/download/models/95263?type=Model&format=Other'
wget -nc -O /workspace/ComfyUI/models/embeddings/BadNegAnatomyV1neg.pt 'https://civitai.com/api/download/models/64063'
wget -nc -O /workspace/ComfyUI/models/embeddings/JuggernautNegativeneg.pt 'https://civitai.com/api/download/models/86553?type=Negative&format=Other'

# Download VAE

mkdir -p /workspace/ComfyUI/models/vae
wget -nc -O /workspace/ComfyUI/models/vae/sdxl_vae.safetensors 'https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors?download=true'
wget -nc -O /workspace/ComfyUI/models/vae/vae-ft-mse-840000-ema-pruned.safetensors 'https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors?download=true'
wget -nc -O /workspace/ComfyUI/models/vae/clearvae_v23.safetensors 'https://civitai.com/api/download/models/88156?type=Model&format=SafeTensor'
wget -nc -O /workspace/ComfyUI/models/vae/anythingKlF8Anime2VaeFtMse840000_klF8Anime2.safetensors 'https://civitai.com/api/download/models/131654'
wget -nc -O /workspace/ComfyUI/models/vae/klF8Anime2VAE_klF8Anime2VAE.safetensors 'https://civitai.com/api/download/models/28569?type=Model&format=SafeTensor'
wget -nc -O /workspace/ComfyUI/models/vae/difconsistencyRAWVAE_v10.safetensors 'https://civitai.com/api/download/models/94036?type=Model&format=PickleTensor'
wget -nc -O /workspace/ComfyUI/models/vae/stage_a.safetensors 'https://huggingface.co/stabilityai/stable-cascade/resolve/main/stage_a.safetensors?download=true'
wget -c https://huggingface.co/WarriorMama777/OrangeMixs/resolve/main/VAEs/orangemix.vae.pt -P ./models/vae/
wget -c https://huggingface.co/hakurei/waifu-diffusion-v1-4/resolve/main/vae/kl-f8-anime2.ckpt -P ./models/vae/


# Download ComfyUI-AnimateDiff-Evolved model files
mkdir -p /workspace/ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved/models
wget -nc -O /workspace/ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved/models/animatediffMotion_sdxlV10Beta.ckpt 'https://civitai.com/api/download/models/219642'
wget -nc -O /workspace/ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved/models/mm-Stabilized_high.pth 'https://huggingface.co/manshoety/AD_Stabilized_Motion/resolve/main/mm-Stabilized_high.pth?download=true'
wget -nc -O /workspace/ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved/models/mm-p_0.75.pth 'https://huggingface.co/manshoety/beta_testing_models/resolve/main/mm-p_0.75.pth?download=true'
wget -nc -O /workspace/ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved/models/temporaldiff-v1-animatediff.safetensors 'https://huggingface.co/CiaraRowles/TemporalDiff/resolve/main/temporaldiff-v1-animatediff.safetensors?download=true'
wget -nc -O /workspace/ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved/models/mm_sd_v14.ckpt 'https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v14.ckpt?download=true'
wget -nc -O /workspace/ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved/models/mm_sd_v15.ckpt 'https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15.ckpt?download=true'
wget -nc -O /workspace/ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved/models/mm_sd_v15_v2.ckpt 'https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt?download=true'

# Download motion lora model files
mkdir -p /workspace/ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved/motion_lora
wget -nc -O /workspace/ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved/motion_lora/v2_lora_PanLeft.ckpt 'https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_PanLeft.ckpt?download=true'
wget -nc -O /workspace/ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved/motion_lora/v2_lora_PanRight.ckpt 'https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_PanRight.ckpt?download=true'
wget -nc -O /workspace/ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved/motion_lora/v2_lora_Stretch.ckpt 'https://huggingface.co/guoyww/animatediff/resolve/main/v2_lora_Stretch.ckpt?download=true'
