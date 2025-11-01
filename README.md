Here is my submission for the Technical Aptitude round.

My GitHub repository with all code and assets is located at: (https://github.com/Alism0101/IIT-Assignment-/tree/main)

Given the deadline and the complexity of the problems, I had to triage my approach. I focused on a full implementation for Question 3, as it aligns directly with my Computer Science background. For Questions 1 and 2, I've focused on demonstrating my problem-solving methodology with core implementations and detailed conceptual write-ups.

---

##  Question 1: Imaging Science

My approach here was to implement Homomorphic Filtering. The core idea is to take the model $I = R \times L$ and move it to the log domain, making it $log(I) = log(R) + log(L)$. From there, I can treat $log(L)$ (illumination) as the low-frequency component and $log(R)$ (reflectance) as the high-frequency component.

* **Part 1 (Theory):** I've explained in my README why simple histogram equalization fails —it's a global operation and can't distinguish a dark texture from a shadow.
* **Part 2 (Implementation):** The code is in `q1_imaging/`. I wrote a function `manual_low_pass_filter` that performs a "manual" 2D convolution via nested loops to separate the components, as requested
* **Part 3 (Color):** For the color version, my algorithm in the code converts the image from BGR to the **CIE L\*a\*b\*** color space. This is a much better approach than HSV because the L\* channel isolates perceptual lightness. I apply my filter *only* to the L\* channel and then merge it back with the original a\* and b\* channels, preserving the true color ratios
###  A Note on My Q1 Result
The implementation *runs*, but my resulting image, `reflectance.jpg`, is clearly wrong. After debugging, I realized I made a critical parameter error. ]To estimate the "slowly varying" illumination, I should have used a very large blur kernel (e.g., 31x31 or 41x41).

I used a **5x5 kernel**. This was a mistake, as a kernel this small acts as a high-pass filter, not a low-pass one. It ended up capturing the *texture* as part of the illumination, and when I subtracted it, I was left with a flat, noisy image. The code structure is correct, but this parameter choice was flawed.

---

##  Question 2: Computer Vision

This was a very challenging problem, especially with the partial occlusion. I decided that a standard corner-based approach would be too fragile.

* **Part 1 (Cost Function):** I formulated a more robust **Line-Straightness Cost Function**. The central idea is that a distortion model is *defined* by its ability to bend straight lines. Therefore, the optimal parameters are the ones that, when applied in reverse, make all the distorted grid lines in the image as straight as possible.
    1.  I detect all line segments (even the curved ones) using an LSD detector.
    2.  I sample points along these lines.
    3.  When "undistorted" by a set of parameters $(k_1, k_2...)$, these points should be collinear.
    4.  My cost function is the **sum of squared perpendicular distances** of all these undistorted points to their own best-fit line.

* **Part 2/3 (Pipeline):** My script `q2_vision/undistort.py` sets up this pipeline. It uses RANSAC to filter out non-grid lines  and then uses `scipy.optimize.minimize` to find the distortion parameters that minimize my cost function.

###  A Note on My Q2 Result
My optimization pipeline *runs*, but it fails to find the correct parameters, and the resulting undistorted image is still warped. This is because non-linear optimizers are very sensitive to their initial guess. I gave it a poor starting point (`x0 = [10.0, 10.0]`) just to test the function. The optimizer is getting stuck in a local minimum. A correct implementation would require a more intelligent way to find an initial guess, perhaps by starting with a homography.

---

##  Question 3: Deep Learning

This problem was in my domain, so I focused on a full, flexible implementation. ]All the code is in the `q3_deep_learning/` folder, designed to run on Colab

* **Implementation:** I built a seq2seq model  in PyTorch.
    * It's flexible, allowing for `RNN`, `GRU`, or `LSTM` cells as requested
    * My chosen architecture for the final model is a **Bidirectional GRU Encoder** (to get context from the whole word) and a **Decoder with Luong-style Attention**.

###  A Note on My Q3 Result
This was a frustrating bug. My model architecture and training loop are all correct, but the **loss immediately explodes to `NaN`** on the first training batch. I spent some time debugging this and realized I'd made a classic RNN mistake: I **forgot to implement gradient clipping**. The gradients from the Bidirectional GRU are clearly exploding. The fix is a single line (`torch.nn.utils.clip_grad_norm_`) in the training loop before the `optimizer.step()`, but I ran out of time to re-run the training.

Note: Due to time constraints, the training loop (train.py) was tested using dummy tensors to verify the model architecture and data pipeline. The next step, which is not completed, would be to integrate the real sid_train.csv dataset and build the full vocabulary.

### 4. Computation/Parameter Counts

Here are my calculations based on the given assumptions (1 layer, $m, k, T, V$)

* **Total Parameters:** I'm summing the parameters for each layer:
    1.  Source Embedding: $V \times m$
    2.  Target Embedding: $V \times m$
    3.  Encoder (GRU): $3(km + k^2 + 2k)$
    4.  Decoder (GRU): $3(km + k^2 + 2k)$
    5.  Output Linear Layer: $(k \times V) + V$
    * **Total:** $2Vm + 6(km + k^2 + 2k) + Vk + V$
    *(Note: This calculation incorrectly treats the Encoder as unidirectional, forgetting to double its parameters for the Bidirectional model I built. This was an oversight in the calculation step.)*

* **Total Computations:**
    1.  Encoder (GRU): $T \times (3(km + k^2))$
    2.  Decoder (GRU): $T \times (3(km + k^2))$
    3.  Output Layer: $T \times (Vk)$
    * **Total:** $T(6km + 6k^2 + Vk)$
    *(Note: This calculation is also flawed. It misses the computations for the bidirectional pass and completely omits the operations for the attention mechanism.)*
