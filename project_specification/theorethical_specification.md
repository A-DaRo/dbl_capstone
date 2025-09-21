# Theoretical Specification

This document outlines the theoretical foundations, design principles, and high-level goals of the Coral-MTL project. It delves into the "why" behind our architectural and methodological choices, framing the project within the broader context of computer vision and ecological research. For a detailed guide on the technical implementation, class structures, and code, please see the [**Technical Specification**](./technical_specification.md).

---

### 1. The Task: A Hierarchical Approach to Semantic Segmentation

The overarching goal of this project is the automated, pixel-level understanding of underwater coral reef imagery for ecological monitoring. Traditional semantic segmentation in this domain often treats all classes as a single, flat list, which fails to capture the inherent relationships and varying levels of importance among them. Our approach re-frames the problem as a **hierarchical multi-task learning (MTL) challenge**. This paradigm is built on the understanding that not all information is of equal value and that the model can benefit from explicitly learning a structured, hierarchical representation of the scene.

The core hypothesis is that by forcing the model to learn simpler, context-providing tasks alongside complex, high-value tasks, we can improve the performance of the latter. The auxiliary tasks act as a form of structured regularization, guiding the shared feature encoder to learn representations that are more robust, disentangled, and context-aware.

#### 1.1. Primary Tasks: The Core Ecological Objectives
These are the central outputs of our model, providing the most critical information for marine biologists and reef managers. They represent the "end goals" of the analysis and demand the highest fidelity.

*   **Genus Segmentation:** This is a fine-grained shape and morphology recognition task. The model must learn to differentiate between the distinct structural forms of various coral genera. This is a challenging "few-shot" problem, as many genera are visually similar and some are rare in the dataset. Success in this task requires the model to move beyond simple color and texture and learn high-level spatial and structural features. For example, it must distinguish the intricate, branching patterns of *Acropora*, the massive, boulder-like shapes of *Porites*, and the flat, tabular structures of *Table Acropora*.

*   **Health Segmentation:** This is primarily a colorimetric and textural analysis task, but with significant morphological dependencies. The model must classify coral pixels into one of three critical health states based on their appearance:
    *   **Healthy/Live:** Characterized by vibrant, natural pigmentation, indicating the presence of symbiotic zooxanthellae.
    *   **Bleached:** Characterized by a stark white appearance due to the expulsion of the algae, a key indicator of thermal stress.
    *   **Dead:** Characterized by a dull color, often covered in turf algae or sediment, but still retaining the underlying coral skeleton structure. Recognizing this requires understanding the underlying shape.

#### 1.2. Auxiliary Tasks: Contextual Understanding and Noise Regularization
These tasks are not primary ecological outputs but are crucial for improving the performance and robustness of the primary tasks. They are designed to explicitly model and disentangle the main sources of "noise" and contextual ambiguity in the underwater survey images. By forcing the model to dedicate capacity to these simpler objectives, we prevent it from becoming distracted by them when trying to solve the primary tasks.

*   **Fish Segmentation:** Models dynamic occluders. Fish frequently obscure parts of the coral. By explicitly teaching the model to identify "fish," we encourage the shared encoder to learn features that can infer the shape of a coral even when it is partially hidden. This is a form of learned in-painting.

*   **Human-Artifact Segmentation:** Models survey-related objects. This task merges several classes (`Human`, `Transect Tools`, `Trash`) into a single category. It teaches the model to identify objects that are not part of the natural reef, preventing them from being confused with corals or other benthic categories.

*   **Substrate Segmentation:** Models the surrounding ecological context. This task identifies key non-coral seabed types like `Sand` and `Seagrass`. This provides the model with powerful contextual priors. For example, learning that certain corals rarely grow on sand can help the model avoid generating false positives in those areas.

---

### 2. The Dataset and Pre-processing: From Flat Labels to Hierarchical Structure

Our model is trained on the **Coralscapes dataset**, a comprehensive collection of high-resolution underwater images with dense, pixel-level annotations for 39 distinct benthic classes. A core theoretical decision is to **not** use these 39 classes directly in a "flat" segmentation model. Such an approach would suffer from extreme class imbalance and would fail to leverage the semantic hierarchy.

Instead, we implement a **Label Transformation Pipeline**. This is a deterministic pre-processing step that converts the monolithic 39-class label space into a structured set of ground-truth masks, one for each of our defined tasks. This transformation is the foundational step that enables our entire hierarchical multi-task learning approach. It recasts a single, complex problem into a set of smaller, more focused, and semantically meaningful sub-problems.

The specific mappings are detailed in the [Technical Specification](./technical_specification.md#21-the-label-transformation-pipeline).

---

### 3. The Model Architecture: A Hierarchical Context-Aware MTL Framework

Our model, inspired by SegFormer and MTLSegFormer, is a novel Transformer-based architecture designed for explicit, context-aware information exchange between tasks. It consists of a shared encoder and an asymmetric, multi-stream decoder.

*   **Shared Encoder:** A single **SegFormer Mix Transformer (MiT) encoder** serves as the backbone. Its role is to process the input RGB image and extract a rich, shared representation of visual features at multiple scales. The use of a shared encoder is central to MTL; it allows the model to learn a generalized feature set that benefits all tasks, and it is computationally efficient. The hierarchical nature of the MiT encoder, producing feature maps at different resolutions, is particularly well-suited for segmentation.

*   **Asymmetric Decoder:** After the shared encoder, the model branches into parallel streams with decoders of varying complexity, reflecting the defined hierarchy of our tasks.
    *   **Primary Streams (Genus & Health):** These are equipped with full, multi-layer **All-MLP decoders**. This provides them with high capacity, allowing them to perform the complex reasoning and spatial refinement required for the primary objectives.
    *   **Auxiliary Streams (Fish, Human-Artifact, Substrate):** These use **lightweight decoder heads** (e.g., a single convolutional layer). This is a deliberate design choice. Since their role is to provide context and act as regularizers, they do not require the same level of expressive power. This asymmetry saves computational resources and helps focus the model's capacity on the primary tasks.

*   **The Core Innovation: Expanded Cross-Attention:** The hub for information exchange is an attention module integrated exclusively within the two Primary Streams. This module allows the Genus and Health tasks to **explicitly query the contextual information** generated by all other tasks. For example, the Genus decoder can ask, "Given the features from the Health, Fish, and Substrate streams, how should I refine my own feature representation?" This allows for a dynamic, learned information flow, where the model can decide which context is most relevant for each pixel.

The detailed implementation of the architecture can be found in the [Technical Specification](./technical_specification.md#3-the-model-architecture-implementation).

---

### 4. Data Sampling Strategy: Context-Aware Spatial Sampling

A major challenge in the Coralscapes dataset is the severe spatial imbalance: vast areas of uninformative background (sand, water) surround small, information-rich coral patches. A naive random patch cropping strategy would be highly inefficient, as the model would waste most of its capacity learning to identify "sand."

To overcome this, we adopt a **Context-Aware Spatial Sampling** strategy, based on the principles of **Poisson Disk Sampling (PDS)**.

*   **Objective:** To generate a high-quality, static training dataset where every patch is guaranteed to contain objects of interest, while also ensuring the sampling covers the full orthomosaic area without excessive clustering or overlap. This ensures the model's training time is spent on information-rich data.

*   **Mechanism:** PDS is an algorithm that produces samples that are tightly packed but no two samples are closer than a specified minimum distance, creating a more uniform, "blue noise" distribution. We seed the algorithm using a foreground mask of all coral classes.

*   **Context-Aware Refinement:** We introduce a crucial refinement where the minimum sampling distance `r` is adapted based on the local class density. In regions with sparse, minority coral genera, `r` is made smaller to allow for more frequent sampling. This ensures that rare classes are adequately represented in the final training pool.

The technical implementation, including optimizations like parallel processing and JIT compilation, is described in the [Technical Specification](./technical_specification.md#4-data-sampling-implementation).

---

### 5. Sample Augmentation

To improve model generalization and prevent overfitting, each patch drawn from the data loader is passed through a sequence of on-the-fly data augmentations. These are divided into two categories, applied sequentially.

*   **Geometric Augmentations:** These teach the model spatial invariance (e.g., to rotation, scaling, and flipping). They are applied to both the image and the corresponding mask stack to maintain their alignment.
*   **Colorimetric Augmentations:** These are applied **only to the input image patch**. They simulate the challenging and variable underwater lighting conditions (e.g., changes in color, brightness, water clarity/turbidity). By not applying these to the masks, we teach the model that the semantic meaning of a pixel is invariant to these lighting changes.

A full list of applied augmentations is available in the [Technical Specification](./technical_specification.md#5-sample-augmentation-pipeline).

---

### 6. Optimizer and Learning Rate

*   **Optimizer:** We use the **AdamW optimizer**. Unlike standard Adam, it decouples the weight decay from the gradient-based updates. For models with high-frequency gradients like Transformers, this has been shown to lead to better generalization and more effective regularization.

*   **Learning Rate Scheduler:** We employ a **Polynomial Decay ("Poly") Learning Rate Scheduler** with a linear warm-up phase. This strategy is critical for stabilizing the training of large Transformer models. The initial warm-up phase allows the model to adapt to the data with small updates before the learning rate increases to its target value, preventing early divergence. The subsequent slow, polynomial decay allows for fine-tuning as training progresses.

Specific hyperparameters are listed in the [Technical Specification](./technical_specification.md#6-optimizer-configuration).

---

### 7. Loss Function

The loss function is a composite, multi-part objective that reflects the hierarchical nature of our tasks and is designed to handle class imbalance and promote clean boundaries.

*   **Total Loss:** A weighted sum of the primary and auxiliary task losses: `L_total = L_primary + w_aux * L_auxiliary`. The `w_aux` hyperparameter ensures the auxiliary tasks contribute as regularizers without overpowering the primary objectives.

*   **Primary Task Loss:** This component uses **uncertainty-based weighting** to automatically balance the Genus and Health tasks. It introduces two learnable parameters, `σ_genus` and `σ_health`, which represent the model's confidence in each task. The model learns to down-weight the loss of the task with higher uncertainty (i.e., the noisier or more difficult task), preventing it from dominating the gradient. The loss for each primary task is a **hybrid of Focal Loss and Dice Loss**:
    *   **Focal Loss:** Addresses the severe class imbalance *within* each task (e.g., common vs. rare genera) by down-weighting the loss for well-classified examples, forcing the model to focus on hard-to-classify ones.
    *   **Dice Loss:** Directly optimizes for spatial overlap (IoU), which is highly effective at combating imbalance and producing spatially coherent, non-fragmented predictions.

*   **Auxiliary Task Loss:** This is a simpler sum of **Weighted Cross-Entropy** losses for the auxiliary tasks. A simpler loss is sufficient as their purpose is regularization, not high-fidelity output.

The precise formulas are detailed in the [Technical Specification](./technical_specification.md#7-loss-function-implementation).

---

### 8. Evaluation Metrics

To rigorously assess our model, we employ a multi-faceted evaluation strategy. Relying on a single metric like mIoU can be misleading.

*   **Primary Task Metrics:**
    *   **Mean Intersection over Union (mIoU):** The gold standard for overall segmentation accuracy.
    *   **Boundary IoU (BIoU):** Specifically measures the quality of predicted boundaries. This is a direct quantitative measure of the "nitid shapes" objective and helps distinguish a model that gets the location right but the boundary wrong.

*   **Overall Model Performance:** We define a custom **Hierarchical Mean (H-Mean)**, the average of the primary task mIoUs (`(mIoU_Genus + mIoU_Health) / 2`), to serve as the single key metric for model selection during validation.

*   **Diagnostic Error Analysis:** Inspired by the TIDE framework, we decompose the total error into distinct, actionable categories calculated from the confusion matrix:
    *   **Classification Error:** A foreground pixel is predicted as the wrong foreground class (e.g., *Acropora* confused with *Pocillopora*).
    *   **Background Error (False Positive):** A background pixel is predicted as any foreground class (e.g., "hallucinating" coral on sand).
    *   **Missed Error (False Negative):** A foreground pixel is predicted as background (e.g., failing to detect a small coral).
    This provides deep, actionable insights into *why* a model fails, guiding future improvements.

The formulas and their implementation context are in the [Technical Specification](./technical_specification.md#8-evaluation-metrics-implementation).

---

### 9. Possible Improvements and Extendability

This section outlines potential future directions to enhance the project's capabilities and robustness.

#### 9.1. Theoretical Enhancements
*   **Self-Supervised Pre-training:** Instead of relying solely on ImageNet pre-trained weights (which are trained on terrestrial images), we could perform self-supervised pre-training (e.g., using Masked Autoencoders or DINO) on a large corpus of unlabeled underwater imagery. This would allow the encoder to learn features more specific to the underwater domain, potentially leading to better fine-tuning performance.
*   **Temporal Analysis:** The current model is static. A significant extension would be to incorporate temporal data, analyzing video sequences or repeated surveys of the same reef over time. This could be achieved with architectures like Video-MAE or by adding temporal attention mechanisms to the current model, enabling tasks like tracking coral growth, disease progression, or recovery from bleaching events.
*   **Advanced Attention Mechanisms:** While cross-attention is powerful, exploring more advanced or efficient attention mechanisms (e.g., linear attention, performer) within the decoder could further improve information flow between tasks or reduce the computational cost, allowing for larger models or higher resolution inputs.
*   **Unsupervised Domain Adaptation (UDA):** The model is trained on a specific dataset. To make it more generalizable to different reef locations, camera systems, or lighting conditions, UDA techniques could be investigated. This would involve training the model on labeled source data and unlabeled target data simultaneously, encouraging the model to learn domain-invariant features.

#### 9.2. New Task Integration
The hierarchical framework is inherently extensible. New tasks can be added to provide more granular information or context.
*   **Disease Segmentation:** A new primary or auxiliary task could be added to specifically identify different types of coral diseases (e.g., black band disease, white plague), which often have subtle visual cues.
*   **Algae Type Segmentation:** Differentiating between different types of algae (e.g., turf algae, macroalgae, coralline algae) could provide more nuanced ecological context than the current substrate task.
*   **Invertebrate Segmentation:** Adding a task to identify other key invertebrates (e.g., sea urchins, crown-of-thorns starfish) could provide valuable data on reef health and threats.

---
For technical details, see the [**Technical Specification**](./technical_specification.md).
