# Adding Conditional Control to Text-to-Image Diffusion Models
https://arxiv.org/pdf/2302.05543
## Abstract
대용량 이미지로 pretrained된 text to image diffusion model에 공간적 조건 제어 기능을 추가하는 모델입니다. canny map, depth, segmentation, Human pose 등과 같은 다양한 조건 제어 방식을 Stable Diffusion 모델에서 단일 또는 다중 조건으로, 프롬프트를 사용하거나 사용하지 않는 방식으로 테스트합니다. 

## Introduction

controlnet은 입력조건을 학습하기 위해 대규모 데이터셋을 학습한 diffusion model을 제어하는 모델입니다.

허용가능한 시간과 메모리 내에서 특정 task에 대해 대형모델의 빠른 학습을 위해서 최적화하는 fine-tuning이나 transfer learning이 필요하다.

diffusion model중 stable diffusion model과 사용하는 것에 대해 제안하고 있다.

## Method
Controlnet은 대규모 데이터셋으로 사전학습된 text to image diffusion 모델에 조건을 추가할 수 있는 구조입니다.
![img](github_page/he.png)




- **c 벡터인 조건(condition)을 추가**

이를 통해 anime 스타일, canny map, depth, open pose 등의 다양한 조건을 추가하여 diffusion 모델을 제어할 수 있다. 

- **회색 블럭의 locked**

이는 copy 파라미터를 고정하여 gradient 연산을 요구하지 않아 연산 복잡도를 효율적으로 할 수 있어 학습 속도와 GPU 메모리 사용량을 줄일 수 있다.

- **zero convolution layer**

가우시안 가중치와 bias가 0으로 초기화된 1x1 convolution layer. 

이를 사용함으로써 학습과정에 있어 노이즈를 방지. 

고품질의 이미지로 예측 (프롬프트에 의존하지 않고 입력 이미지 정보를 최대한 반영). 

- **trainable copy**

원래의 가중치를 직접 학습하는 대신 사본을 만들어 데이터 셋이 작을 때의 과적합을 방지.

대규모 데이터셋으로 사전학습된 모델의 성능을 최대한 유지할 수 있다는 장점. 

### FAQ

**Q:** But wait, if the weight of a conv layer is zero, the gradient will also be zero, and the network will not learn anything. Why "zero convolution" works?

**A:** This is not true. [See an explanation here](docs/faq.md).

# Stable Diffusion + ControlNet

By repeating the above simple structure 14 times, we can control stable diffusion in this way:

![img](github_page/sd.png)

In this way, the ControlNet can **reuse** the SD encoder as a **deep, strong, robust, and powerful backbone** to learn diverse controls. Many evidences (like [this](https://jerryxu.net/ODISE/) and [this](https://vpd.ivg-research.xyz/)) validate that the SD encoder is an excellent backbone.

Note that the way we connect layers is computational efficient. The original SD encoder does not need to store gradients (the locked original SD Encoder Block 1234 and Middle). The required GPU memory is not much larger than original SD, although many layers are added. Great!

# Features & News

2023/0/14 - We released [ControlNet 1.1](https://github.com/lllyasviel/ControlNet-v1-1-nightly). Those new models will be merged to this repo after we make sure that everything is good.

2023/03/03 - We released a discussion - [Precomputed ControlNet: Speed up ControlNet by 45%, but is it necessary?](https://github.com/lllyasviel/ControlNet/discussions/216)

2023/02/26 - We released a blog - [Ablation Study: Why ControlNets use deep encoder? What if it was lighter? Or even an MLP?](https://github.com/lllyasviel/ControlNet/discussions/188)

2023/02/20 - Implementation for non-prompt mode released. See also [Guess Mode / Non-Prompt Mode](#guess-anchor).

2023/02/12 - Now you can play with any community model by [Transferring the ControlNet](https://github.com/lllyasviel/ControlNet/discussions/12).

2023/02/11 - [Low VRAM mode](docs/low_vram.md) is added. Please use this mode if you are using 8GB GPU(s) or if you want larger batch size.

# Production-Ready Pretrained Models

First create a new conda environment

    conda env create -f environment.yaml
    conda activate control

All models and detectors can be downloaded from [our Hugging Face page](https://huggingface.co/lllyasviel/ControlNet). Make sure that SD models are put in "ControlNet/models" and detectors are put in "ControlNet/annotator/ckpts". Make sure that you download all necessary pretrained weights and detector models from that Hugging Face page, including HED edge detection model, Midas depth estimation model, Openpose, and so on. 

We provide 9 Gradio apps with these models.

All test images can be found at the folder "test_imgs".

## ControlNet with Canny Edge

Stable Diffusion 1.5 + ControlNet (using simple Canny edge detection)

    python gradio_canny2image.py

The Gradio app also allows you to change the Canny edge thresholds. Just try it for more details.

Prompt: "bird"
![p](github_page/p1.png)

Prompt: "cute dog"
![p](github_page/p2.png)

## ControlNet with M-LSD Lines

Stable Diffusion 1.5 + ControlNet (using simple M-LSD straight line detection)

    python gradio_hough2image.py

The Gradio app also allows you to change the M-LSD thresholds. Just try it for more details.

Prompt: "room"
![p](github_page/p3.png)

Prompt: "building"
![p](github_page/p4.png)

## ControlNet with HED Boundary

Stable Diffusion 1.5 + ControlNet (using soft HED Boundary)

    python gradio_hed2image.py

The soft HED Boundary will preserve many details in input images, making this app suitable for recoloring and stylizing. Just try it for more details.

Prompt: "oil painting of handsome old man, masterpiece"
![p](github_page/p5.png)

Prompt: "Cyberpunk robot"
![p](github_page/p6.png)

## ControlNet with User Scribbles

Stable Diffusion 1.5 + ControlNet (using Scribbles)

    python gradio_scribble2image.py

Note that the UI is based on Gradio, and Gradio is somewhat difficult to customize. Right now you need to draw scribbles outside the UI (using your favorite drawing software, for example, MS Paint) and then import the scribble image to Gradio. 

Prompt: "turtle"
![p](github_page/p7.png)

Prompt: "hot air balloon"
![p](github_page/p8.png)

### Interactive Interface

We actually provide an interactive interface

    python gradio_scribble2image_interactive.py

~~However, because gradio is very [buggy](https://github.com/gradio-app/gradio/issues/3166) and difficult to customize, right now, user need to first set canvas width and heights and then click "Open drawing canvas" to get a drawing area. Please do not upload image to that drawing canvas. Also, the drawing area is very small; it should be bigger. But I failed to find out how to make it larger. Again, gradio is really buggy.~~ (Now fixed, will update asap)

The below dog sketch is drawn by me. Perhaps we should draw a better dog for showcase.

Prompt: "dog in a room"
![p](github_page/p20.png)

## ControlNet with Fake Scribbles

Stable Diffusion 1.5 + ControlNet (using fake scribbles)

    python gradio_fake_scribble2image.py

Sometimes we are lazy, and we do not want to draw scribbles. This script use the exactly same scribble-based model but use a simple algorithm to synthesize scribbles from input images.

Prompt: "bag"
![p](github_page/p9.png)

Prompt: "shose" (Note that "shose" is a typo; it should be "shoes". But it still seems to work.)
![p](github_page/p10.png)

## ControlNet with Human Pose

Stable Diffusion 1.5 + ControlNet (using human pose)

    python gradio_pose2image.py

Apparently, this model deserves a better UI to directly manipulate pose skeleton. However, again, Gradio is somewhat difficult to customize. Right now you need to input an image and then the Openpose will detect the pose for you.

Prompt: "Chief in the kitchen"
![p](github_page/p11.png)

Prompt: "An astronaut on the moon"
![p](github_page/p12.png)

## ControlNet with Semantic Segmentation

Stable Diffusion 1.5 + ControlNet (using semantic segmentation)

    python gradio_seg2image.py

This model use ADE20K's segmentation protocol. Again, this model deserves a better UI to directly draw the segmentations. However, again, Gradio is somewhat difficult to customize. Right now you need to input an image and then a model called Uniformer will detect the segmentations for you. Just try it for more details.

Prompt: "House"
![p](github_page/p13.png)

Prompt: "River"
![p](github_page/p14.png)

## ControlNet with Depth

Stable Diffusion 1.5 + ControlNet (using depth map)

    python gradio_depth2image.py

Great! Now SD 1.5 also have a depth control. FINALLY. So many possibilities (considering SD1.5 has much more community models than SD2).

Note that different from Stability's model, the ControlNet receive the full 512×512 depth map, rather than 64×64 depth. Note that Stability's SD2 depth model use 64*64 depth maps. This means that the ControlNet will preserve more details in the depth map.

This is always a strength because if users do not want to preserve more details, they can simply use another SD to post-process an i2i. But if they want to preserve more details, ControlNet becomes their only choice. Again, SD2 uses 64×64 depth, we use 512×512.

Prompt: "Stormtrooper's lecture"
![p](github_page/p15.png)

## ControlNet with Normal Map

Stable Diffusion 1.5 + ControlNet (using normal map)

    python gradio_normal2image.py

This model use normal map. Rightnow in the APP, the normal is computed from the midas depth map and a user threshold (to determine how many area is background with identity normal face to viewer, tune the "Normal background threshold" in the gradio app to get a feeling).

Prompt: "Cute toy"
![p](github_page/p17.png)

Prompt: "Plaster statue of Abraham Lincoln"
![p](github_page/p18.png)

Compared to depth model, this model seems to be a bit better at preserving the geometry. This is intuitive: minor details are not salient in depth maps, but are salient in normal maps. Below is the depth result with same inputs. You can see that the hairstyle of the man in the input image is modified by depth model, but preserved by the normal model. 

Prompt: "Plaster statue of Abraham Lincoln"
![p](github_page/p19.png)

## ControlNet with Anime Line Drawing

We also trained a relatively simple ControlNet for anime line drawings. This tool may be useful for artistic creations. (Although the image details in the results is a bit modified, since it still diffuse latent images.)

This model is not available right now. We need to evaluate the potential risks before releasing this model. Nevertheless, you may be interested in [transferring the ControlNet to any community model](https://github.com/lllyasviel/ControlNet/discussions/12).

![p](github_page/p21.png)

<a id="guess-anchor"></a>

# Guess Mode / Non-Prompt Mode

The "guess mode" (or called non-prompt mode) will completely unleash all the power of the very powerful ControlNet encoder. 

See also the blog - [Ablation Study: Why ControlNets use deep encoder? What if it was lighter? Or even an MLP?](https://github.com/lllyasviel/ControlNet/discussions/188)

You need to manually check the "Guess Mode" toggle to enable this mode.

In this mode, the ControlNet encoder will try best to recognize the content of the input control map, like depth map, edge map, scribbles, etc, even if you remove all prompts.

**Let's have fun with some very challenging experimental settings!**

**No prompts. No "positive" prompts. No "negative" prompts. No extra caption detector. One single diffusion loop.**

For this mode, we recommend to use 50 steps and guidance scale between 3 and 5.

![p](github_page/uc2a.png)

No prompts:

![p](github_page/uc2b.png)

Note that the below example is 768×768. No prompts. No "positive" prompts. No "negative" prompts.

![p](github_page/uc1.png)

By tuning the parameters, you can get some very intereting results like below:

![p](github_page/uc3.png)

Because no prompt is available, the ControlNet encoder will "guess" what is in the control map. Sometimes the guess result is really interesting. Because diffusion algorithm can essentially give multiple results, the ControlNet seems able to give multiple guesses, like this:

![p](github_page/uc4.png)

Without prompt, the HED seems good at generating images look like paintings when the control strength is relatively low:

![p](github_page/uc6.png)

The Guess Mode is also supported in [WebUI Plugin](https://github.com/Mikubill/sd-webui-controlnet):

![p](github_page/uci1.png)

No prompts. Default WebUI parameters. Pure random results with the seed being 12345. Standard SD1.5. Input scribble is in "test_imgs" folder to reproduce.

![p](github_page/uci2.png)

Below is another challenging example:

![p](github_page/uci3.png)

No prompts. Default WebUI parameters. Pure random results with the seed being 12345. Standard SD1.5. Input scribble is in "test_imgs" folder to reproduce.

![p](github_page/uci4.png)

Note that in the guess mode, you will still be able to input prompts. The only difference is that the model will "try harder" to guess what is in the control map even if you do not provide the prompt. Just try it yourself!

Besides, if you write some scripts (like BLIP) to generate image captions from the "guess mode" images, and then use the generated captions as prompts to diffuse again, you will get a SOTA pipeline for fully automatic conditional image generating.

# Combining Multiple ControlNets

ControlNets are composable: more than one ControlNet can be easily composed to multi-condition control.

Right now this feature is in experimental stage in the [Mikubill' A1111 Webui Plugin](https://github.com/Mikubill/sd-webui-controlnet):

![p](github_page/multi2.png)

![p](github_page/multi.png)

As long as the models are controlling the same SD, the "boundary" between different research projects does not even exist. This plugin also allows different methods to work together!

# Use ControlNet in Any Community Model (SD1.X)

This is an experimental feature.

[See the steps here](https://github.com/lllyasviel/ControlNet/discussions/12).

Or you may want to use the [Mikubill' A1111 Webui Plugin](https://github.com/Mikubill/sd-webui-controlnet) which is plug-and-play and does not need manual merging.

# Annotate Your Own Data

We provide simple python scripts to process images.

[See a gradio example here](docs/annotator.md).

# Train with Your Own Data

Training a ControlNet is as easy as (or even easier than) training a simple pix2pix. 

[See the steps here](docs/train.md).

# Related Resources

Special Thank to the great project - [Mikubill' A1111 Webui Plugin](https://github.com/Mikubill/sd-webui-controlnet) !

We also thank Hysts for making [Hugging Face Space](https://huggingface.co/spaces/hysts/ControlNet) as well as more than 65 models in that amazing [Colab list](https://github.com/camenduru/controlnet-colab)! 

Thank haofanwang for making [ControlNet-for-Diffusers](https://github.com/haofanwang/ControlNet-for-Diffusers)!

We also thank all authors for making Controlnet DEMOs, including but not limited to [fffiloni](https://huggingface.co/spaces/fffiloni/ControlNet-Video), [other-model](https://huggingface.co/spaces/hysts/ControlNet-with-other-models), [ThereforeGames](https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/7784), [RamAnanth1](https://huggingface.co/spaces/RamAnanth1/ControlNet), etc!

Besides, you may also want to read these amazing related works:

[Composer: Creative and Controllable Image Synthesis with Composable Conditions](https://github.com/damo-vilab/composer): A much bigger model to control diffusion!

[T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models](https://github.com/TencentARC/T2I-Adapter): A much smaller model to control stable diffusion!

[ControlLoRA: A Light Neural Network To Control Stable Diffusion Spatial Information](https://github.com/HighCWu/ControlLoRA): Implement Controlnet using LORA!

And these amazing recent projects: [InstructPix2Pix Learning to Follow Image Editing Instructions](https://www.timothybrooks.com/instruct-pix2pix), [Pix2pix-zero: Zero-shot Image-to-Image Translation](https://github.com/pix2pixzero/pix2pix-zero), [Plug-and-Play Diffusion Features for Text-Driven Image-to-Image Translation](https://github.com/MichalGeyer/plug-and-play), [MaskSketch: Unpaired Structure-guided Masked Image Generation](https://arxiv.org/abs/2302.05496), [SEGA: Instructing Diffusion using Semantic Dimensions](https://arxiv.org/abs/2301.12247), [Universal Guidance for Diffusion Models](https://github.com/arpitbansal297/Universal-Guided-Diffusion), [Region-Aware Diffusion for Zero-shot Text-driven Image Editing](https://github.com/haha-lisa/RDM-Region-Aware-Diffusion-Model), [Domain Expansion of Image Generators](https://arxiv.org/abs/2301.05225), [Image Mixer](https://twitter.com/LambdaAPI/status/1626327289288957956), [MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation](https://multidiffusion.github.io/)

# Citation

    @misc{zhang2023adding,
      title={Adding Conditional Control to Text-to-Image Diffusion Models}, 
      author={Lvmin Zhang and Anyi Rao and Maneesh Agrawala},
      booktitle={IEEE International Conference on Computer Vision (ICCV)}
      year={2023},
    }

[Arxiv Link](https://arxiv.org/abs/2302.05543)

[Supplementary Materials](https://lllyasviel.github.io/misc/202309/cnet_supp.pdf)
