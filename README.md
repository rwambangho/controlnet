# Adding Conditional Control to Text-to-Image Diffusion Models
https://arxiv.org/pdf/2302.05543
## Abstract
대용량 이미지로 pretrained된 text to image diffusion model에 공간적 조건 제어 기능을 추가하는 모델입니다. canny map, depth, segmentation, Human pose 등과 같은 다양한 조건 제어 방식을 Stable Diffusion 모델에서 단일 또는 다중 조건으로, 프롬프트를 사용하거나 사용하지 않는 방식으로 테스트합니다. 

## Introduction

controlnet은 입력조건을 학습하기 위해 대규모 데이터셋을 학습한 diffusion model을 제어하는 모델입니다.

허용가능한 시간과 메모리 내에서 특정 task에 대해 대형모델의 빠른 학습을 위해서 최적화하는 fine-tuning이나 transfer learning이 필요합니다.

저자는 diffusion model중 stable diffusion model과 사용하는 것에 대해 제안하고 있습니다.

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


![](https://velog.velcdn.com/images/bh9711/post/6196e41b-8a7d-4889-8570-2fba37bd91bf/image.png)


- Stable Diffusion 구조

stable diffusion의 U-net 구조는 controlnet의 encoder블럭과 middle블럭이 연결되어 있습니다. 

고정된 회색 블럭은 stable diffusion이며 모두 25개의 블럭으로 구성되있으며 encoder와 decoder가 모두 12개이고, 이 블럭 중 8개의 블럭은 다운샘플링 또는 업샘플링 conv layer를 뜻합니다. 나머지 17개의 블럭은 4개의 resnet 레이어와 2개의 vit를 포함하는 기본블럭입니다. 각 vit에는 몇가지 cross attention과 self attention 매커니즘이 적용되어 있습니다. (SD Encoder Block은 Resnet레이어 4개와 vit 2개로 구성되어있고 x3은 블럭을 3번 반복 한다는 의미) middle 블럭이 controlnet의 trainable copy인 새로운 파라미터를 받는 encoder 블럭과 연결되어 있습니다. 

텍스트는 OpenAI CLIP으로 인코딩되고 diffusion timestep은 위치 인코딩으로 인코딩됩니다.

 

Stable Diffusion은 VQ-GAN과 유사한 전처리 방법을 사용하여 안정화된 학습을 위해 512×512 이미지의 전체 데이터셋을 더 작은 64×64 크기의 latent 이미지로 변환합니다. 이를 위해서는 convolution 크기와 일치하도록 이미지 기반 조건들을 64×64 feature space로 변환하는 ControlNet이 필요합니다.

- controlnet 구조

12개 블럭으로 되어 있는 trainable copy와 stable diffusion의 middle 블럭 1개를 사용하며 4개의 해상도를 가지는 3개의 블럭으로 되어있습니다. 12개의 skip connection 과 middle블럭을 지나 출력이 나옵니다. 전형적인 u-net구조로 되어있기 때문에 다른 diffusion모델과 적용해도 된다고 합니다. zero conv layer는 가중치와 bias가 0인 1x1 conv layer를 뜻하는데 stable Diffusion의 decoder블럭과 연결됩니다.(concat) 

이게 무엇을 뜻하냐면 zero convolution은 학습을 통해 0에서 최적화된 파라미터로 점진적으로 성장하는 고유한 유형의 연결 레이어가 됩니다.

이 방식으로 진행할시 copy 파라미터가 고정되어 있고 gradient computation이 요구되지 않기 때문에 효율적인 연산복잡도를 가집니다. 

![](https://velog.velcdn.com/images/bh9711/post/7ecf5661-8d73-4f23-89bc-5d68f6555119/image.png)


x=input feature map, F(x; Θ)=neural network블럭, Z(·; ·)=zero conv layer, Θz1 and Θz2= zero conv parameter, yc=controlnet블럭의 최종 output

- Training과정

![](https://velog.velcdn.com/images/bh9711/post/eb8a83b2-e8bf-4817-987d-0c4f9642c829/image.png)


loss계산은 기존 stable diffusion과 동일합니다. 

 zt=노이즈 이미지, t=타임스텝, ct=텍스트 프롬프트, cf=입력하는 조건들

모델은 t시점에 추가된 노이즈를 예측하고 예측된 노이즈와 t시점에 실제로 추가된 노이즈와의 차를 구해서 loss로 사용합니다.

controlnet의 학습과정 중 입력되는 텍스트 프롬프트의 경우 학습과정에서는 50%는 랜덤하게 빈 문자열로 변환되어 학습됩니다. 이는 프롬프트의 정보를 제한하고 인풋되는 조건에서 의미론적인 내용들을 학습하고자 하는 방향으로 학습하는 것입니다.

sudden convergence phenomenon이 관찰(갑작스러운 수렴 현상) 

주어진 조건들이 잘 반영된 이미지들을 모델이 갑자기 생성하는 현상으로 보통 1만optimization step 이전쯤에 갑작스러운 수렴 현상이 관찰된다고 합니다.

아래 사과 이미지의 경우 6100 step 까지는 안 나오다가 1만optimization step 이전인 6133 step 에서 갑자기 잘 나오게 되는 것을 확인할 수 있습니다.

![](https://velog.velcdn.com/images/bh9711/post/deb36263-c407-4209-b66c-7849dcaed05c/image.png)


zero conv가 최적화되지 않다 하더라도 기존 stable diffusion의 파라미터를 freeze 즉, copy해서 사용하고 있기 때문에 고품질의 이미지를 생성할 수 있습니다. 6100까진 학습이 최적화되지 않더라도 이미 freeze에서 사용하고 있는 파라미터에 의해서 고품질의 이미지가 생성됨을 확인할 수 있고 조건이 반영되지 않은 고품질의 이미지를 생성하다가 zero conv의 최적화가 진행되면서 조건을 잘 이해할 정도로 최적화가 되었을때 조건이 반영된 고품질의 이미지를 갑자기 생성해내는 것입니다.

- Inference

![](https://velog.velcdn.com/images/bh9711/post/b752253e-cc7a-43f4-95d5-ec2a5f62869b/image.png)


denoising diffusion 프로세스에서 controlnet에 주어진 추가적인 조건 정보들을 제어할 수 있는 방법에 대해 소개하고 있습니다.  기존 stable diffusion 추론과정에서는 classifier free guidance 라는 개념을 사용합니다. 

앱실론prd=최종output, 앱실론uc= 조건정보를 반영하지 않은 ouput, 앱실론c= 조건정보를 사용해서 도출하는 output, 베타cfg =guidance scale(조건정보를 얼마나 반영하여서 이미지를 생성할 것인가를 제어할 수 있는 사용자 지정 가중치)

classifier free guidance 베타값을 통해서 stable diffusion은 얼마나 이 조건정보를 반영하여 이미지를 생성할 수 있을지 제어할 수 있는 것. 
classifier free guidance 통해 실험을 진행. 
맨 좌측에 input conditioning이미지. 프롬프트가 없는 상태로 실험을 진행.

### 실험 결과

CGF-scale:9.0

sampler: DDIM

여러가지 Task에 대한 실험을 진행했다고 합니다. 

그 중 스케치 이미지 condition과 4가지 프롬프트 세팅에 대한 실험결과에 대해 확인해보았습니다.

![](https://velog.velcdn.com/images/bh9711/post/4a5b5c5c-53a1-4c34-9fb8-852ac0bb22c4/image.png)


각각 모델 구조를 변형하여 진행

a=논문에서 제안한 구조

b=zero conv를 Gaussian가중치로 초기화된 standard conv로 대체

c=Trainable copy대신 단일conv로 변경

첫번째 열은 프롬프트를 빈 문자열로 생성한 것

두번째 열은 정보가 부족한 텍스트

세번째 열은 무관한 텍스트

네번쨰 열은 완벽한 텍스트

프롬프트를 잘 입력할 수록 좋은 성능을 보였습니다.

Trainable copy와 zero conv의 효과가 강력한것으로 보이며 zero conv가 없거나 Trainable copy를 사용하지 않은 구조의 경우 Trainable copy가 가지고 있었던 사전학습된 정보들이 파인튜닝 과정에서 destroy되어 이미지 생성을 제대로 못하고 있는 것을 알 수 있었습니다.

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



