+++
title = "Diffusion Models for Dummies: Getting Started"
date = 2026-05-09T12:42:42+05:30
tags = ["Diffusion", "Deep Learning"]
description = "An Intuitive Explanation of Diffusion Models with a hands-on Implementation"
+++ 




Generative AI has been the buzz and face of AI industry for the last few years. Particularly image and video generative models. How do they models generative an image when given a prompt? But how exactly does a model take in text prompt and transform it into a visual output catering to the context of that prompt?

This is where diffusion comes in ! One of the most amazing model architectures and training methodologies in modern AI.

My intention with this blog will be to give you a **intuitive** overview of some of the math and the training workflow of diffusion along with a **hands-on implementation** at the end. While I will go over the necessary math involved, it does not even start to cover the entirety of this beautiful topic. However, it should be good enough to get a rough idea of the underlying principles and should(hopefully) motivate you to dive deeper.

Here are the topics that we will follow:
- Images as Samples of a distribution
- A Model as an Approximation for a Distribution
- Diffusion - Forward Process - Generating Training Data
- Diffusion - Reverse Process - Generating Image from Noise
- Implementation
---
## Images belong to a Distribution
Since diffusion based models rely on the idea of transforming a probability distributions and then sampling from them, it is judicious to first understand how images can be seen as samples from some probability distribution.

Any image is just a grid of pixels and each pixel is just a tuple of RGB values. If you consider an image to be a set of variables then you can create any image of resolution HxW with **3xHxW independent continuous variables** ranged between 0-255(or some other common scale). For a typical image generator, generating the image is like sampling from a **joint probability distribution** of all these variables. 

Ofcourse, if the distribution is just some standard distribution like the Gaussian then you will just get a noisy incoherent image. But given that the distribution is a relevant and useful one , the kind of images you want will **have high probability densities** and the useless irrelevant ones will have lower.

During training also we like to assume that all the images of our image dataset belong to some joint probability distribution over the variables  and we train our model to learn this ideal distribution from the training data points.

<figure class="my-8">
  <img src="/blogs/diffusion_tut/distributions.png" alt="data distribution" class="w-full rounded-sm border border-gray-200 dark:border-gray-800/70">
  <figcaption class="text-center text-sm text-gray-600 dark:text-gray-400 mt-3"><i>In this figure, the "good" images of dogs have high values of the pdf whereas the irrelevant images(like the cookie) have low values. Ideally, we would want our trained model to be able to capture these high density regions and generate only the good high probability images accordingly.</i></figcaption>
</figure>


---
## Neural Networks as an Approximation
It is very difficult to arrive at a closed form solution for getting this probability distribution given the very high dimensionality of the sample space. Hence we use neural networks to approximate this distribution function.

For a discrete sample space, for example in the case of classification problems, we have fixed set of classes into which we need to segregate an image. So typically what we do is we train the model to take in an input and give a probability mass function over the different classes. Then we choose the class with the highest probability as the identified class of the input.

However in the case of image generation we are dealing with a continuous space  and so the approximated function has to be a probability density function over this sample space. How can this be done?

One naive approach would be to try training a model to give a real value for each possible image pixel combination which sums to 1 over the entire space. But there is no way to ensure the summation property other than to divide the ouput by the sum/integral Z over the sample space; which is computationally infeasible for even moderately sized sample spaces.

$$
Z = \int_{x} f(x) dx \quad\quad p(x)=\frac{1}{Z} f(x) $$

The general practice that is followed and works out well is to use NNs to approximate the target distributions using **already known probability distributions** and using the NN to approximate their defining parameters(mean and variance). For instance, the most standard practice is to use the ***Gaussian Distribution*** due to a lot of good properties, which I won't discuss here.

Therefore, we define the network as  
$$p_\theta(x) = \mathcal{N}(x; \mu_\theta, \Sigma_\theta)
$$
A neural network model with parameters $\theta$ that approximates the $\mu$ and $\Sigma$ of the required multivariate normal distribution.


---
## Diffusion for Image Generation

### Denoising Reverse Process

Now in diffusion we don't exactly find the required probability distribution directly. Instead we follow an *iterative process* of slowly transforming a starting **noisy distribution** into the required final distribution. Why is this done? A simple explanation would be, it just works better for the practical scenarios of approximating non-linear functions in high dimensions.

The theoretical diffusion sampling process (the generative Markov chain) is defined by Gaussian transitions 
$$
p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
$$
but in practice we consider a only time-dependent variance $\Sigma_\theta(x_t, t) = \Sigma_t$. Learning the variance is mathematically "heavy" and can make training unstable. Early research found that simply following the same "amount" of noise used in the forward process works remarkably well.
$$
p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_t)
$$
At each step the model takes the previous noisy image $x_t$ and calculates the mean for the previous image $x_{t-1}$ and then samples $x_{t-1}$ from the distribution $\mathcal{N}(\mu_\theta(x_t, t), \Sigma_t)$. 

Now, we don't ask the neural network to spit out $\mu$ directly because that's a difficult learning task.

Instead we use use something called **reparameterisation trick** to get the values. We use the neural network to predict a noise amount $\epsilon_\theta$ and then use that to approximate the $\mu_{\theta}$ value. 
$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)$$
Notice that the model takes in input $x_t$ and t; we will get back to this a bit later. The alpha and beta values are just fixed values depending on the noising schedule that is being followed. 

*Why do we do this reparameterisation?* -  Because it makes the training much **more stable** - similar to how teaching ResNets to predict the change in input is easier for the model than predicting the output estimate. If you are curious why then refer to [this paper](https://arxiv.org/pdf/1712.09913).
<!-- ![alt text](image-4.png)
![alt text](image-3.png) -->

<figure class="my-8">
  <img src="/blogs/diffusion_tut/markov_chain.png" alt="diffusion markov chain" class="w-full rounded-sm border border-gray-200 dark:border-gray-800/70">
  <figcaption class="text-center text-sm text-gray-600 dark:text-gray-400 mt-3"><i>This chain illustrates how we go from a noisy image sample to a more structured one over time steps</i></figcaption>
</figure>

Now given a trained model with parameters $\theta$ , the sampling algorithm for generating an image from the target distribution is as follows:

1. **Initialize:** Start with $x_T \sim \mathcal{N}(0, \mathbf{I})$ (pure white noise).
2. **Iterate:** For $t = T, T-1, \dots, 1$:
    - **Sample random noise:** $z \sim \mathcal{N}(0, \mathbf{I})$ if $t > 1$, else $z = 0$.
    - **Predict the noise:** Pass the current image $x_t$ and the time step $t$ into your model to get $\epsilon_\theta(x_t, t)$.
    - **Compute the Mean** ($\mu_\theta$): This is the "de-noised" version of the current image.$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)$$
    - **Update the Image:** Calculate the next step $x_{t-1}$ by adding back a tiny bit of controlled variance ($\sigma_t z$) to keep the process stochastic:$$x_{t-1} = \mu_\theta(x_t, t) + \sigma_t z$$
3. **Result:** $x_0$ is your generated image.

This **reverse** process is summarised in the image below. The *sample in red* is slowly transformed to a sample from the target distribution over multiple timesteps.

<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">

<div class="diffusion-reverse-playground">
  <div class="reverse-layout">
    <div class="reverse-visual-side">
      <div class="canvas-slider-container">
        <div class="slider-track-wrapper">
          <span class="axis-label top-label">$t = T$</span>
          <input type="range" id="reverse-slider" min="0" max="100" value="100" step="1" orient="vertical">
          <span class="axis-label bottom-label">$t = 0$</span>
        </div>
        <canvas id="reverse-canvas" width="400" height="300"></canvas>
      </div>
    </div>
    <div class="reverse-control-side">
      <h3>Reverse Diffusion: Skewed Modes</h3>
      <p>Samples($x_T$) from the initial standard Gaussian distribution(top) get slowly transformed into samples($x_0$) of the target distribution(bottom) over $T$ steps.</p>
      <p style="font-size: 0.6rem; color: grey; font-style: italic;">(Please reload the page if object not working properly)</p>
      <div class="status-badge">
        <span id="reverse-step-display">Timestep $t$: 100</span>
      </div>
    </div>
  </div>
</div>

<style>
.diffusion-reverse-playground {
  font-family: var(--font-sans, sans-serif);
  max-width: 780px;
  margin: 1.5rem auto;
  padding: 1.2rem;
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 12px;
  background: rgba(255, 255, 255, 1);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.06);
}

.reverse-layout {
  display: flex;
  gap: 1.5rem;
  align-items: center;
}

.reverse-visual-side {
  flex-shrink: 0;
  background: rgba(0, 0, 0, 0.02);
  padding: 12px;
  border-radius: 10px;
  border: 1px solid rgba(0, 0, 0, 0.05);
}

.canvas-slider-container {
  display: flex;
  flex-direction: row;
  align-items: center;
  gap: 12px;
}

.slider-track-wrapper {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  align-items: center;
  position: relative;
  width: 35px;
}

#reverse-slider {
  -webkit-appearance: slider-vertical;
  appearance: slider-vertical;
  width: 8px;
  height: 220px;
  margin: auto 0;
  accent-color: #4f46e5;
  cursor: pointer;
}

.axis-label {
  font-size: 0.7rem;
  font-weight: 600;
  color: #64748b;
}

#reverse-canvas {
  border-radius: 6px;
  display: block;
  background: transparent;
}

.reverse-control-side {
  flex-grow: 1;
  text-align: left;
}

.reverse-control-side h3 {
  margin: 0 0 0.4rem 0;
  color: #0f172a;
  font-size: 1.15rem;
  font-weight: 600;
  letter-spacing: -0.01em;
}

.reverse-control-side p {
  font-size: 0.85rem;
  color: #475569;
  line-height: 1.5;
  margin: 0 0 1.2rem 0;
}

.status-badge {
  display: inline-block;
  background: rgba(79, 70, 229, 0.1);
  padding: 4px 10px;
  border-radius: 20px;
}

#reverse-step-display {
  color: #4f46e5;
  font-weight: 600;
  font-size: 0.85rem;
}

@media (max-width: 640px) {
  .reverse-layout {
    flex-direction: column;
    text-align: center;
  }
  .reverse-control-side {
    text-align: center;
    width: 100%;
  }
}
</style>

<script>
function initReverseSimulation() {
  const revCanvas = document.getElementById('reverse-canvas');
  if (!revCanvas || revCanvas.dataset.initialized) return;
  revCanvas.dataset.initialized = 'true';

  const revCtx = revCanvas.getContext('2d');
  const revSlider = document.getElementById('reverse-slider');
  const revDisplay = document.getElementById('reverse-step-display');

  const NUM_PARTICLES = 45; 
  const MAX_STEPS = 100;
  const trajectories = [];
  const W = revCanvas.width;
  const H = revCanvas.height;

  function getGaussian() {
    let u = 0, v = 0;
    while(u === 0) u = Math.random(); 
    while(v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }

  for (let p = 0; p < NUM_PARTICLES; p++) {
    const path = new Array(MAX_STEPS + 1);
    const targetLeft = Math.random() < 0.75;
    
    let currentX = (W / 2) + getGaussian() * 30;
    path[MAX_STEPS] = currentX;

    for (let t = MAX_STEPS - 1; t >= 0; t--) {
      let drift = 0;
      if (t < 75) {
        const targetX = targetLeft ? (W / 2 - 95) : (W / 2 + 95);
        drift = (targetX - currentX) * 0.045; 
      }
      const noiseScale = 2.4 * Math.sqrt(t / MAX_STEPS + 0.1);
      currentX += drift + getGaussian() * noiseScale;
      path[t] = currentX;
    }
    
    trajectories.push({
      points: path,
      isRed: p === 7
    });
  }

  function gaussianPDF(x, mean, stdDev) {
    return (1 / (stdDev * Math.sqrt(2 * Math.PI))) * Math.exp(-Math.pow(x - mean, 2) / (2 * Math.pow(stdDev, 2)));
  }

  function drawSimulation() {
    revCtx.clearRect(0, 0, W, H);
    
    const currentStep = parseInt(revSlider.value);
    revDisplay.textContent = `Timestep t: ${currentStep}`;

    const paddingY = 40;
    const usableHeight = H - (paddingY * 2);
    
    function getCanvasY(stepValue) {
      const ratio = (MAX_STEPS - stepValue) / MAX_STEPS;
      return paddingY + ratio * usableHeight;
    }

    const activeY = getCanvasY(currentStep);

    revCtx.lineWidth = 1.75;
    revCtx.strokeStyle = 'rgba(15, 23, 42, 0.25)'; // Darkened slightly for clearer contrast
    
    // Top single distribution (t=T) - Amplified multiplier to 2000
    revCtx.beginPath();
    for (let x = 0; x < W; x++) {
      let yCurve = paddingY - (gaussianPDF(x, W / 2, 30) * 2000);
      if (x === 0) revCtx.moveTo(x, yCurve); else revCtx.lineTo(x, yCurve);
    }
    revCtx.stroke();

    // Bottom skewed bimodal distribution (t=0) - Amplified multiplier to 2600
    revCtx.beginPath();
    for (let x = 0; x < W; x++) {
      let pLeft = gaussianPDF(x, W / 2 - 95, 22) * 0.75;
      let pRight = gaussianPDF(x, W / 2 + 95, 22) * 0.25;
      
      let yCurve = (H - paddingY) - ((pLeft + pRight) * 2600);
      if (x === 0) revCtx.moveTo(x, yCurve); else revCtx.lineTo(x, yCurve);
    }
    revCtx.stroke();

    // Active Timestamp Axis
    revCtx.beginPath();
    revCtx.strokeStyle = 'rgba(0, 0, 0, 0.05)';
    revCtx.setLineDash([4, 4]);
    revCtx.moveTo(0, activeY);
    revCtx.lineTo(W, activeY);
    revCtx.stroke();
    revCtx.setLineDash([]); 

    // Render Particles & Trails
    trajectories.forEach(tr => {
      if (tr.isRed) {
        revCtx.strokeStyle = '#ef4444';
        revCtx.lineWidth = 2.5;
      } else {
        revCtx.strokeStyle = 'rgba(59, 130, 246, 0.2)';
        revCtx.lineWidth = 1.1;
      }

      revCtx.beginPath();
      revCtx.moveTo(tr.points[MAX_STEPS], getCanvasY(MAX_STEPS));
      for (let s = MAX_STEPS - 1; s >= currentStep; s--) {
        revCtx.lineTo(tr.points[s], getCanvasY(s));
      }
      revCtx.stroke();

      revCtx.beginPath();
      revCtx.fillStyle = tr.isRed ? '#ef4444' : '#3b82f6';
      revCtx.arc(tr.points[currentStep], activeY, tr.isRed ? 4.5 : 2.5, 0, 2 * Math.PI);
      revCtx.fill();
      if (tr.isRed) {
        revCtx.strokeStyle = '#ffffff';
        revCtx.lineWidth = 1;
        revCtx.stroke();
      }
    });
  }

  revSlider.addEventListener('input', drawSimulation);
  drawSimulation();
}
setInterval(initReverseSimulation, 200);
</script>


<!-- <div class="flex flex-col sm:flex-row gap-4 sm:gap-6 my-8 items-start">
  <figure class="w-full sm:w-[48%] m-0">
    <img src="/blogs/diffusion_tut/prior_distribution.png" alt="prior distribution" class="w-full rounded-sm border border-gray-200 dark:border-gray-800/70">
    <figcaption class="text-center text-sm text-gray-600 dark:text-gray-400 mt-3"><i>Target Distribution of the data.</i></figcaption>
  </figure>
  <figure class="w-full sm:w-[52%] m-0">
    <img src="/blogs/diffusion_tut/transformation.png" alt="distribution transformation" class="w-full rounded-sm border border-gray-200 dark:border-gray-800/70">
    <figcaption class="text-center text-sm text-gray-600 dark:text-gray-400 mt-3"><i>Samples(see the red one) from the initial Gaussian distribution are slowly tranformed to samples of the target distribution</i></figcaption>
  </figure>
</div> -->



This algorithm essentially starts with a single sample from a random distribution and then iteratively transforms the distribution to the target distribution over T steps such that the sample gets transformed to a sample from the target distribution and since in the target distribution only the coherent samples have high probabilities , the final sample should also be coherent and desirable.
<!-- Note:
- **Variance Schedule ($\alpha, \bar{\alpha}, \beta$):** These must be the exact same values used during the training (Forward) phase.
- **Total Steps ($T$):** The number of iterations (e.g., 1000). -->

---
### The Forward Process and Training
Now that you understand(hopefully) the process of generation of an image via diffusion, we can discuss how these models can be trained.

While there has been a multitude of training methods catering to diffusion and similar architectures, I will be going over the most well-known one which was discussed in the foundational paper of DDPM.

Since we were using the trained model to predict the noise of a given image + timestamp input, we will need to create data to serve this purpose and train it accordingly with some loss function.  How do we make the training data for this? We make use of any image dataset and start adding noise at different intensities to the given image!

But we don't do it in any arbitrary manner, rather we do a time-scheduled addition, hence the "t" in the model input.

In the forward process, we gradually add Gaussian noise to an image step by step. We control how much noise is added at each step using a variance schedule $\beta_t$. If we define $\alpha_t = 1 - \beta_t$, the single-step transition from $x_{t-1}$ to $x_t$ is defined as:

$$x_t = \sqrt{\alpha_t} x_{t-1} + \sqrt{1 - \alpha_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

If we had to do this iteratively to generate a noisy image at step $t$, it would be very slow during training. Fortunately, we can recursively expand this formula to sample an image $x_t$ at any arbitrary timestep $t$ *directly* from the clean image $x_0$, without iterating through all previous steps:

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

Where:
- $\bar{\alpha_t} = \prod_{i=1}^t \alpha_i$ (the cumulative product of the noise schedule)
- $\epsilon$ is the "ground truth" noise we injected.
    
<div class="diffusion-playground">
  <div class="playground-layout">
    <div class="visual-side">
      <img id="source-img" src="https://picsum.photos/id/1025/300/300" crossorigin="anonymous" alt="Source" style="display:none;">
      <canvas id="diffusion-canvas" width="300" height="300"></canvas>
    </div>
    <div class="control-side">
      <h3>Interactive Forward Diffusion</h3>
      <p>Drag the slider to add Forward Gaussian Noise ($q(x_t | x_{t-1})$) to the original image data.</p>
      <div class="controls">
        <div class="slider-labels">
          <span>Original ($x_0$)</span>
          <span id="step-display">Timestep $t$: 0</span>
          <span>Pure Noise ($x_T$)</span>
        </div>
        <input type="range" id="noise-slider" min="0" max="100" value="0" step="1">
      </div>
    </div>
  </div>
</div>
<style>
.diffusion-playground {
  font-family: var(--font-sans, sans-serif);
  max-width: 680px; /* Wider container */
  margin: 1.5rem auto;
  padding: 1.2rem; /* Smaller padding */
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 12px;
  /* Translucent background blur effect */
  background: rgba(255, 255, 255, 1);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.06);
}
.playground-layout {
  display: flex;
  gap: 1.5rem;
  align-items: center;
}
.visual-side {
  flex-shrink: 0;
  display: flex;
  background: rgba(0, 0, 0, 0.04);
  padding: 6px;
  border-radius: 10px;
}
#diffusion-canvas {
  border-radius: 6px;
  display: block;
  width: 180px; /* Slightly scaled down visual weight */
  height: 180px;
}
.control-side {
  flex-grow: 1;
  text-align: left;
}
.control-side h3 {
  margin: 0 0 0.4rem 0;
  color: #0f172a;
  font-size: 1.15rem;
  font-weight: 600;
  letter-spacing: -0.01em;
}
.control-side p {
  font-size: 0.85rem;
  color: #475569;
  line-height: 1.45;
  margin: 0 0 1.2rem 0;
}
.controls {
  background: rgba(255, 255, 255, 0.5);
  padding: 0.8rem;
  border-radius: 8px;
  border: 1px solid rgba(0, 0, 0, 0.03);
}
.slider-labels {
  display: flex;
  justify-content: space-between;
  font-size: 0.75rem;
  color: #64748b;
  margin-bottom: 0.5rem;
  font-weight: 500;
}
#step-display {
  color: #4f46e5;
  font-weight: 600;
}
#noise-slider {
  width: 100%;
  display: block;
  margin: 0;
  accent-color: #4f46e5;
  cursor: pointer;
}
/* Responsiveness for mobile screens */
@media (max-width: 560px) {
  .playground-layout {
    flex-direction: column;
    text-align: center;
    gap: 1rem;
  }
  .control-side {
    text-align: center;
    width: 100%;
  }
  #diffusion-canvas {
    width: 220px;
    height: 220px;
  }
}
</style>
<script>
function initForwardPlayground() {
  const canvas = document.getElementById('diffusion-canvas');
  if (!canvas || canvas.dataset.initialized) return;
  canvas.dataset.initialized = 'true';
  const ctx = canvas.getContext('2d');
  const img = document.getElementById('source-img');
  const slider = document.getElementById('noise-slider');
  const stepDisplay = document.getElementById('step-display');
  let originalPixels = null;
  function randomGaussian() {
    let u = 0, v = 0;
    while(u === 0) u = Math.random(); 
    while(v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }
  function applyNoise() {
    if (!originalPixels) return;
    const width = canvas.width;
    const height = canvas.height;
    const imgData = ctx.createImageData(width, height);
    const targetData = imgData.data;
    const srcData = originalPixels.data;
    const t = slider.value / 100; 
    const beta = t; 
    const alphaBar = 1 - beta;
    const sqrtAlphaBar = Math.sqrt(alphaBar);
    const sqrtOneMinusAlphaBar = Math.sqrt(beta);
    for (let i = 0; i < srcData.length; i += 4) {
      for (let c = 0; c < 3; c++) {
        const originalValue = srcData[i + c];
        const noise = randomGaussian() * 127.5 + 127.5;
        let mixedValue = (sqrtAlphaBar * originalValue) + (sqrtOneMinusAlphaBar * noise);
        targetData[i + c] = Math.min(255, Math.max(0, mixedValue));
      }
      targetData[i + 3] = 255; 
    }
    ctx.putImageData(imgData, 0, 0);
    stepDisplay.textContent = `Timestep t: ${slider.value}`;
  }
  img.onload = function() {
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    originalPixels = ctx.getImageData(0, 0, canvas.width, canvas.height);
    applyNoise();
  };
  if (img.complete) {
    img.onload();
  }
  slider.addEventListener('input', applyNoise);
}
setInterval(initForwardPlayground, 200);
</script>


We provide the model (usually a U-Net) with the noisy image $x_t$ and the current timestep $t$. The model's job is to predict the noise $\epsilon$ that was used to corrupt $x_0$. 

What is the most simple loss function you can think of that might be used here? We are predicting a continuous value! This is just regression on the noise values. Therefore, a sane choice would be the **Mean Squared Error ($L_2$ norm)** between the true noise $\epsilon$ and the predicted noise $\epsilon_\theta$: 
$$\mathcal{L}_{simple}(\theta) = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

> #### Why $L_2$?
> Mathematically, minimizing the $L_2$ error in noise prediction is equivalent to maximizing the **Evidence Lower Bound (ELBO)** under the assumption that the reverse transitions are Gaussian. Maximizing the ELBO is what ensures that our approximation of the true distribution is as close as possible to the true distribution.

#### The Training Algorithm:
1. **Sample** a clean image $x_0$ from your dataset.
2. **Pick a random timestep** $t$ between $1$ and $T$ (e.g., $t=450$ out of $1000$).
3. **Generate random noise** $\epsilon$ and mix it with $x_0$ using the formula above to create $x_t$.
4. Pass the noisy image $x_t$ and the timestep $t$ to the model and get the predicted noise $\epsilon_\theta(x_t, t)$.
5. Compare the predicted noise with the actual noise $\epsilon$ using Mean Squared Error ($L_2$ norm) and calculate the loss.
6. **Optimise:** Take a gradient descent step to minimize the loss.
7. Repeat for a number of epochs.

---



## Implementation
First define the hyper-parameters and device. I will be using Pytorch for this implementation. I would suggest using a GPU since training on a CPU would take a lot of time. You may use Google Colab or Kaggle notebooks for this.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms

T = 500  # Total number of diffusion steps
BETA_START = 0.0001
BETA_END = 0.02

# Flowers102 are RGB natural images; use a slightly larger size than MNIST
IMAGE_SIZE = 64
CHANNELS = 3  # 3 for RGB (Flowers102)
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 100
TIME_DIM = 64

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

```

```python
def get_dataloader():
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    root = "./data"
    train_ds = datasets.Flowers102(root, split="train", download=True, transform=transform)
    val_ds = datasets.Flowers102(root, split="val", download=True, transform=transform)
    dataset = ConcatDataset([train_ds, val_ds])

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )
    return dataloader

betas = torch.linspace(BETA_START, BETA_END, T).to(device)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)

if device == "cuda":
    torch.cuda.manual_seed_all(42)
    betas = betas.to(device)
    alphas = alphas.to(device)
    alpha_bars = alpha_bars.to(device)

sqrt_alpha_bars = torch.sqrt(alpha_bars)
sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

dataloader = get_dataloader()
```

Before we can train, we need to pre-calculate our "noise schedule." These $\alpha$ and $\beta$ values are the knobs that control how much noise is added at each step. By calculating the cumulative products ($\bar{\alpha}$) upfront, we can jump from a clean image ($x_0$) to a noisy version at any timestep $t$ in a single step, rather than looping $t$ times.

To keep things simple, I'm using the **Flowers102** dataset. It’s small enough to train relatively quickly but diverse enough to show that the model is actually learning something interesting. We resize everything to 64x64 and normalize the pixels to the range `[-1, 1]`. This is important because our model will be predicting noise that also lives around that range(due to sampling from the standard Gaussian distribution).

<figure>
  <img src="/blogs/diffusion_tut/flower102.png" alt="Flower 102 dataset">
  <figcaption class="text-center text-sm"><i>A few samples from the Flowers 102 dataset.</i></figcaption>
</figure>


```python 
def forward_diffusion_sample(x0, t, device=None):
    if device is None:
        device = x0.device
    t = t.to(device)

    s_alpha_bar = sqrt_alpha_bars.to(device)[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    s_one_minus_alpha_bar = sqrt_one_minus_alpha_bars.to(device)[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    noise = torch.randn_like(x0, device=device)
    xt = s_alpha_bar * x0 + s_one_minus_alpha_bar * noise

    return xt, noise
```
This function is our **Forward Process**. It takes a clean image and a timestep value $t$, then "corrupts" the image by mixing it with random noise at that corresponding timestep $t$ based on the noise schedule $\beta_t$. 


```python
def get_time_embedding(t, dim):
    """Sinusoidal Positional Encoding for time."""
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    sin_emb = torch.sin(t.unsqueeze(-1) * inv_freq.to(t.device))
    cos_emb = torch.cos(t.unsqueeze(-1) * inv_freq.to(t.device))
    emb = torch.cat([sin_emb, cos_emb], dim=-1)
    return emb
```
As mentioned before the model predicts the noise at a given timestep $t$ from the input noised image. Denoising at step $t=999$ (almost pure noise) is very different from denoising at $t=5$ (almost clean). However we don't just pass the timestep $t$ directly as an integer input to the model. 

Instead a good practice is to encode the timestep $t$ into a vector embedding representation. Here we use **Sinusoidal Positional Encodings**—the same stuff used in Transformers—to turn a single integer $t$ into a rich vector that the neural network can understand.


```python
def _pick_gn_groups(channels: int) -> int:
    for g in (32, 16, 8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        g = _pick_gn_groups(channels)
        self.norm1 = nn.GroupNorm(g, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(g, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        h = self.conv1(F.gelu(self.norm1(x)))
        h = self.conv2(F.gelu(self.norm2(h)))
        return x + h
```
For the "brain" of our model, we use **Residual Blocks**. If you've seen ResNets, this looks familiar. The idea is to let the gradient flow easily through the network by adding the input back to the output of the convolutional layers. We also use **Group Normalization** instead of Batch Norm, as it tends to be more stable when working with smaller batch sizes and generative tasks.


```python
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim=TIME_DIM,  base_channels: int = 32):
        super().__init__()

        c1, c2 = base_channels, base_channels * 2

        time_hidden = time_dim * 2
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_hidden),
            nn.GELU(),
            nn.Linear(time_hidden, time_hidden),
            nn.LayerNorm(time_hidden)
        )
        self.time_to_c1 = nn.Linear(time_hidden, c1)
        self.time_to_c2 = nn.Linear(time_hidden, c2)


        # Encoder
        self.enc_in = nn.Conv2d(in_channels, c1, 3, padding=1)
        self.enc1 = ResBlock(c1)
        self.down1 = nn.Conv2d(c1, c2, 4, stride=2, padding=1)  # 28->14
        self.enc2 = ResBlock(c2)

        # Bottleneck
        self.mid = ResBlock(c2)

        # Decoder
        self.up1 = nn.ConvTranspose2d(c2, c1, 4, stride=2, padding=1)  # 14->28
        self.dec1 = nn.Sequential(
            nn.Conv2d(c1 + c1, c1, 3, padding=1),
            ResBlock(c1),
        )
        self.out = nn.Conv2d(c1, out_channels, 3, padding=1)

    def _inject(self, h, time_cond, time_proj):
        h = h + time_proj(time_cond).unsqueeze(-1).unsqueeze(-1) * 0.1
        return h

    def forward(self, x, t):
        time_emb = get_time_embedding(t, TIME_DIM)
        time_cond = self.time_mlp(time_emb)

        # Encoder
        h1 = self.enc1(F.gelu(self.enc_in(x)))
        h1 = self._inject(h1, time_cond, self.time_to_c1)

        h2 = self.enc2(F.gelu(self.down1(h1)))
        h2 = self._inject(h2, time_cond, self.time_to_c2)

        # Bottleneck
        h = self.mid(h2)
        h = self._inject(h, time_cond, self.time_to_c2)

        # Decoder
        h = F.gelu(self.up1(h))
        h = self.dec1(torch.cat([h, h1], dim=1))
        h = self._inject(h, time_cond, self.time_to_c1)

        return self.out(h)
```
The **U-Net** is the gold standard for image-to-image tasks. It first squeezes the image down to a "bottleneck" to learn high-level features, and then expands it back to the original size. The "skip connections" allows the model to retain fine-grained details from the original noisy image while the bottleneck focus on the overall structure. I've also added a small `_inject` helper method to merge our time embeddings into every layer so the model stays aware of the current timestep.


```python
def train_ddpm(model, dataloader, epochs=EPOCHS):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=LEARNING_RATE * 0.1)
    model.train()

    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            x0, _ = batch
            x0 = x0.to(device)

            # 1. Randomly sample timesteps t for the entire batch
            t = torch.randint(0, T, (x0.shape[0],), device=device).long()

            # 2. Get the noisy input (xt) and the actual noise (epsilon)
            xt, noise = forward_diffusion_sample(x0, t, device=device)

            # 3. Predict the noise using the U-Net
            predicted_noise = model(xt, t)

            # 4. Calculate the loss (L = MSE(noise, predicted_noise))
            loss = loss_fn(noise, predicted_noise)

            # 5. Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            if step % 100 == 0:
                print(f"Epoch {epoch}/{epochs} | Step {step}/{len(dataloader)} | Loss: {loss.item():.4f}")
    
        scheduler.step()

        # Optional: Save model checkpoint after each epoch
        # torch.save(model.state_dict(), f"ddpm_mnist_epoch_{epoch}.pth")

    return model
```
The training loop is surprisingly simple once the math is set up. We pick a clean image, pick a random $t$, corrupt the image using our forward function, and ask the U-Net to guess the noise. We then compare the guess to the actual noise using a simple **MSE Loss**. Over time, the U-Net gets better at seeing "through" the noise to identify the underlying patterns.

```python
def sample_reverse_process(model, num_images=1,no_of_transitions=5):

    # Start with pure noise (xT)
    x = torch.randn((num_images, CHANNELS, IMAGE_SIZE, IMAGE_SIZE), device=device)

    model.eval()
    sample_transitions = []  # To store intermediate samples for visualization
    with torch.no_grad():
        # Loop backward from T-1 down to 0
        for t in reversed(range(T)):
            t_tensor = torch.full((num_images,), t, dtype=torch.long, device=device)

            # 1. Predict the noise (epsilon_theta)
            predicted_noise = model(x, t_tensor)

            # 2. DDPM Reverse Diffusion Step Formula (calculates x_{t-1})
            alpha_t = alphas[t]
            alpha_bar_t = alpha_bars[t]

            # Calculate the predicted mean (mu_theta)
            mean_part1 = 1 / torch.sqrt(alpha_t)
            mean_part2 = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
            mu_theta = mean_part1 * (x - mean_part2 * predicted_noise)

            # 3. Add variance/noise for stochasticity (if t > 0)
            if t > 0:
                variance = betas[t]
                sigma = torch.sqrt(variance)
                z = torch.randn_like(x)
                x = mu_theta + sigma * z
            else:
                # At t=0, the final step is deterministic
                x = mu_theta

            if t % (T // no_of_transitions) == 0 or t == T-1:
                sample_transitions.append((x.cpu().clone(), t))  # Store intermediate samples
    # Clamp output to [-1, 1] range
    x = torch.clamp(x, -1., 1.)
    
    return x,sample_transitions
```

Finally, we have the **Reverse Process**. This is where the magic happens. We start with a block of pure, random Gaussian noise and slowly "chip away" at it using our trained model. At each step, the model predicts the noise, we use the math we derived earlier to calculate the "de-noised" mean ($\mu_\theta$), and then we add back a tiny bit of random variance. This randomness is crucial—it's what allows the model to generate a different, unique flower every time even if you started with similar noise.

---
## Results

I trained it for 100 epochs on a RTX 3060 laptop GPU and got a 97% loss reduction. The following are some results generated by unconditional sampling using the trained model:
![training results](/blogs/diffusion_tut/training_results.png)
As you can see the a lot of the results bear resemblance to flowers. Ofcourse with better architectures and more compute the obtained results can be further improved.

<figure>
    <img src="/blogs/diffusion_tut/denoising_result.png" alt="generated samples">
    <figcaption class="text-center text-sm"><i>The intermediate reverse process steps showing how the image is slowly being generated from pure noise.</i></figcaption>
</figure>

<br>

You can access the full notebook with the code [here](https://github.com/Utsab-2010/Blogs-and-Implementations/blob/main/Diffusion/Diffusion_tutorial.ipynb).

Let me know in the comments if any portion needs to be elaborated further. I wanted to discuss conditional generation too but I did not want to drag it further. I shall try to take it up in another blog. I also plan to write another one where we discuss the maths in more detail, deriving the expressions from first principles. 

Thanks for reading :)

---

## Resources
 - [DDPM: Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
 - [Simplest Implementation of Diffusion Models \| Emilio’s Blog](https://e-dorigatti.github.io/math/deep%20learning/2023/06/25/diffusion.html)
 - [Diffusion Models - The AI Summer](https://theaisummer.com/diffusion-models/)
 - [Lecture Notes on Diffusion Models](http://arxiv.org/abs/2312.10393)
- A great course on Generative Diffusion and Flow Models - [YouTube](https://www.youtube.com/playlist?list=PL57nT7tSGAAXwjhDYcxEycx5W7YoSrZyt)
