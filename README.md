## Mini-JSTorch


Mini-JSTorch is a lightweight, `dependency-free` JavaScript neural network library designed for `education`, `experimentation`, and `small-scale models`.
It runs in Node.js and modern browsers, with a simple API inspired by PyTorch-style workflows.

This project prioritizes `clarity`, `numerical correctness`, and `accessibility` over performance or large-scale production use.

In this version `1.8.0`, we Introduce the **SoftmaxCrossEntropyLoss**, and **BCEWithLogitsLoss**

# NOTICE
All source code for `mini-jstorch-github` was pulled from `npmjs.com/package/mini-jstorch` with some minor refactoring.

**WARNING:** Expect potential compatibility issues in certain modules when running in the GitHub environment.

---

# Overview

**Mini-JSTorch provides a minimal neural network engine implemented entirely in plain JavaScript.**

*It is intended for:*

- learning how neural networks work internally
- experimenting with small models
- running simple training loops in the browser
- environments where large frameworks are unnecessary or unavailable

`Mini-JSTorch is NOT a replacement for PyTorch, TensorFlow, or TensorFlow.js.`

`It is intentionally scoped to remain small, readable, and easy to debug.`

---

# Key Characteristics

- Zero dependencies
- Works in Node.js or others enviornments and browser environments
- Explicit, manual forward and backward passes
- Focused on 2D training logic (`[batch][features]`)
- Designed for educational and experimental use

---

# Browser Support

Now, Mini-JSTorch can be used directly in browsers:

- via ESM imports
- via CDN / `<script>` with a global `JST` object

This makes it suitable for:

- demos
- learning environments
- lightweight frontend experiments

Here example code to make a simple Model with JSTorch.
In Browser/Website:

```html
<!DOCTYPE html>
<html>
<body style="font-family:monospace; padding:20px;">
    <h3>mini-jstorch XOR Demo</h3>
    <div id="status">Initializing...</div>
    <pre id="log" style="background:#eee; padding:10px;"></pre>
    <div id="res"></div>

    <script type="module">
        import { Sequential, Linear, ReLU, MSELoss, Adam, StepLR, Tanh } from 'https://unpkg.com/mini-jstorch@1.8.0/index.js';
        
        async function train() {
            const statusEl = document.getElementById('status');
            const logEl = document.getElementById('log');
            
            try {
                const model = new Sequential([
                    new Linear(2, 16), new Tanh(),
                    new Linear(16, 8), new ReLU(),
                    new Linear(8, 1)
                ]);

                const X = [[0,0], [0,1], [1,0], [1,1]];
                const y = [[0], [1], [1], [0]];
                const criterion = new MSELoss();
                const optimizer = new Adam(model.parameters(), 0.1);
                const scheduler = new StepLR(optimizer, 25, 0.5);

                for (let epoch = 0; epoch <= 1000; epoch++) {
                    const loss = criterion.forward(model.forward(X), y);
                    model.backward(criterion.backward());
                    optimizer.step();
                    scheduler.step();
                    
                    if (epoch % 200 === 0) {
                        logEl.textContent += `Epoch ${epoch} | Loss: ${loss.toFixed(6)}\n`;
                        statusEl.textContent = `Training: ${epoch}/1000`;
                        await new Promise(r => setTimeout(r, 1));
                    }
                }

                statusEl.textContent = '✅ Done';
                const preds = model.forward(X);
                document.getElementById('res').innerHTML = `<h4>Results:</h4>` + 
                    X.map((input, i) => `[${input}] -> <b>${preds[i][0].toFixed(4)}</b> (Target: ${y[i][0]})`).join('<br>');

            } catch (e) {
                statusEl.textContent = '❌ Error: ' + e.message;
            }
        }
        train();
    </script>
</body>
</html>
```

---

# Core Features

# Layers

- Linear
- Flatten
- Conv2D (*experimental*)

# Activations

- ReLU
- Sigmoid
- Tanh
- LeakyReLU
- GELU
- Mish
- SiLU
- ELU

# Loss Functions

- MSELoss
- CrossEntropyLoss (*legacy*)
- SoftmaxCrossEntropyLoss (**recommended**)
- BCEWithLogitsLoss (**recommended**)

# Optimizers

- SGD
- Adam
- AdamW
- Lion

# Learning Rate Schedulers

- StepLR
- LambdaLR
- ReduceLROnPlateau
- Regularization
- Dropout (*basic*, *educational*)
- BatchNorm2D (*experimental*)

# Utilities

- zeros
- randomMatrix
- dot
- addMatrices
- reshape
- stack
- flatten
- concat
- softmax
- crossEntropy

# Model Container

- Sequential

---

# Installation 

## Node.js
```bash
npm install mini-jstorch
```
Node.js v18+ or any modern browser with ES module support is recommended.

## Git
```bash
git clone https://github.com/Rizal-HID11/mini-jstorch-github
```

---

# Quick Start (Recommended Loss)

# Multi-class Classification (SoftmaxCrossEntropy)

```javascript
import {
  Sequential,
  Linear,
  ReLU,
  SoftmaxCrossEntropyLoss,
  Adam
} from "./src/jstorch.js";

const model = new Sequential([
  new Linear(2, 4),
  new ReLU(),
  new Linear(4, 2) // logits output
]);

const X = [
  [0,0], [0,1], [1,0], [1,1]
];

const Y = [
  [1,0], [0,1], [0,1], [1,0]
];

const lossFn = new SoftmaxCrossEntropyLoss();
const optimizer = new Adam(model.parameters(), 0.1);

for (let epoch = 1; epoch <= 300; epoch++) {
  const logits = model.forward(X);
  const loss = lossFn.forward(logits, Y);
  const grad = lossFn.backward();
  model.backward(grad);
  optimizer.step();

  if (epoch % 50 === 0) {
    console.log(`Epoch ${epoch}, Loss: ${loss.toFixed(4)}`);
  }
}
```
Do not combine `SoftmaxCrossEntropyLoss` with a `Softmax` layer.

# Binary Classifiaction (BCEWithLogitsLoss)

```javascript
import {
  Sequential,
  Linear,
  ReLU,
  BCEWithLogitsLoss,
  Adam
} from "./src/jstorch.js";

const model = new Sequential([
  new Linear(2, 4),
  new ReLU(),
  new Linear(4, 1) // logit
]);

const X = [
  [0,0], [0,1], [1,0], [1,1]
];

const Y = [
  [0], [1], [1], [0]
];

const lossFn = new BCEWithLogitsLoss();
const optimizer = new Adam(model.parameters(), 0.1);

for (let epoch = 1; epoch <= 300; epoch++) {
  const logits = model.forward(X);
  const loss = lossFn.forward(logits, Y);
  const grad = lossFn.backward();
  model.backward(grad);
  optimizer.step();
}
```
Do not combine `BCEWithLogitsLoss` with a `Sigmoid` layer.

---

# Save & Load Models 

```javascript
import { saveModel, loadModel, Sequential } from "mini-jstorch";

const json = saveModel(model);
const model2 = new Sequential([...]); // same architecture
loadModel(model2, json);
```

---

# Demos

See the `demo/` directory for runnable examples:
- `demo/MakeModel.js` – simple training loop
- `demo/scheduler.js` – learning rate schedulers
- `demo/fu_fun.js` – utility functions

```bash
node demo/MakeModel.js
node demo/scheduler.js
node demo/fu_fun.js
```

---

# Design Notes & Limitations 

- Training logic is 2D-first: `[batch][features]`
- Higher-dimensional data is reshaped internally by specific layers (e.g. Conv2D, Flatten)
- No automatic broadcasting or autograd graph
- Some components (Conv2D, BatchNorm2D, Dropout) are educational / experimental
- Not intended for large-scale or production ML workloads

---

# Intended Use Cases

- Learning how neural networks work internally
- Teaching ML fundamentals
- Small experiments in Node.js or the browser
- Lightweight AI demos without GPU or large frameworks

--- 

# License

MIT License

Copyright (c) 2024
rizal-editors

---
