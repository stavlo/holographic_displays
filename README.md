# DeepDPE - improving double phase encoding using deep neural network

## Agenda
* Project overview
* Environment Setting
* Dataset
* Training
* Results
* Credits and References


### Project overview
Holography is a technique used in optics and imaging that aims to capture and reconstruct 
the phase and the amplitude information of the light which come from an object.
However, current methods often struggle to simultaneously achieve precise per-pixel control
over both the phase and amplitude in holograms
Spatial light modulators (SLMs) have emerged as a promising tool to achieve phase control
while maintaining a constant amplitude. 
In light of this, our project aims to leverage SLM technology to transform the encoded 
information in both the amplitude and phase of light into a phase-only representation.
Our goal is to use deep neural networks to enhance and improve the classical solutions in phase holography.

### Environment Setting
You can install all necessary packages by running the following command:
```bash
pip install -r requirements.txt
```

### Dataset
We tried to encode the next image from Big Buck Bunny movie

![image](https://github.com/stavlo/holographic_displays/blob/net_per_color/datasets/1.png?raw=true)


### Training
During the training process, we implemented four different models:
* Using a classic method call double phase encoding (DPE) 
* Using DPE initialized with convolution network
* Using DPE initialized with convolution network with non-linear activation and skip connection
* Trie to learn from scratch using the amplitude and phase 

You can train the model using `main.py`, choosing different hyperparameters and training modes:
```bash
python main.py \
--epochs, default=200
--batch_size, default=1
--optimizer, default="adam"
--lr, default=1e-4
--z, default=0.1, help='[m]'
--wave_length, default=np.asfarray([638 * 1e-9, 520 * 1e-9, 450 * 1e-9]), help='[m]'
--eval, default=False
--overfit, default=True
--model, default='conv', '[conv, skip_connection, classic, amp_phs]'
--Loss, default='[TV_loss]', '[TV_loss, L1, L2, perceptual_loss, laplacian_kernel]'
```

### results
We conducted experiments with different distance and models, each expirement out put is:
* Loss graph
* Output image
* Set of weight for the best loss

#### Example for graph of the train and validation loss:
????????????????????????
#### Example for an output image:
??????????????
### Credits and References
We based our project on the results of the following papers and Github repositories:
* [Neural 3D Holography] (http://www.computationalimaging.org/publications/neuralholography3d/) (Suyeon Choi, et al. 2021)
* [Tensor Holography V2] (http://cgh-v2.csail.mit.edu/) (Liang Shi, et al. 2022)
* [Perceptual Loss](https://github.com/pytorch/examples/blob/7f7c222b355abd19ba03a7d4ba90f1092973cdbc/fast_neural_style/neural_style/neural_style.py#L55
): Vgg16 perceptual loss network
* Perceptual Loss] (https://arxiv.org/pdf/1603.08155.pdf) (Justin Johnson, et al. 2016)


