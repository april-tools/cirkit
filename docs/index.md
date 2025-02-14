---
hide:
    - toc
    - navigation
    - title
---

<div class="grid" markdown>


<div markdown>
# Building efficient and trustworthy AI
**cirkit** is a framework for building, learning and reasoning about **probabilistic machine learning** models, such as [circuits](https://arxiv.org/abs/2409.07953) and [tensor networks](https://arxiv.org/abs/1708.00006), which are **tractable** and **expressive**.
</div>


* ⚡ **Exact and Efficient Inference** : Automatically compile tractable models to efficient computational graphs that run on the GPU.
* **Compatible**: Seamlessly integrate your circuit with deep learning models; run on any device compatible with PyTorch.
* **Modular and Extensible**: Support for user-defined layers and parameterizations that extend the symbolic language of cirkit.
* **Templates for Common Cases**: Templates for constructing circuits by mixing layers and structures with a few lines of code.


</div>

<h1>Getting Started</h1>

<div class="grid cards" markdown>

-   :fontawesome-solid-screwdriver-wrench:{ .lg .middle } __Build a circuit...__

    ---
    [:octicons-arrow-right-24: from region graphs](notebooks/region-graphs-and-parametrisation.ipynb)

<!--

    [:octicons-arrow-right-24: with different layers :fontawesome-solid-layer-group:{.lg}](#)

--->

-   :fontawesome-solid-gears:{ .lg .middle } __Learn a circuit...__

    ---
    [:octicons-arrow-right-24: for distribution estimation :fontawesome-solid-chart-area:{.lg}](notebooks/learning-a-circuit.ipynb)

   	[:octicons-arrow-right-24: for tensor compression :fontawesome-solid-file-zipper:](notebooks/compression-cp-factorization.ipynb)

    [:octicons-arrow-right-24: as a (generative) multi-class classifier](notebooks/generative-vs-discriminative-circuit.ipynb)

    [:octicons-arrow-right-24: ... all of the above, with PICs :fontawesome-solid-camera:{.lg}](notebooks/learning-a-circuit-with-pic.ipynb)

-   :material-scale-balance:{ .lg .middle }__Advanced reasoning...__

    ---
    [:octicons-arrow-right-24: with squared circuits $($:fontawesome-solid-plug-circle-minus:{.lg}$)^2$](notebooks/sum-of-squares-circuits.ipynb)

    [:octicons-arrow-right-24: with logic circuits :fontawesome-solid-square-binary:{.lg}...](notebooks/logic-circuits.ipynb)

	</br>
    ...to enforce constraints in neural nets

<!--
-   :fontawesome-solid-code-merge:{ .lg .middle } __Integrate with other PyTorch libraries...__

    ---
    [:octicons-arrow-right-24: ZUKO: normalising flows](#)
--->

</div>
