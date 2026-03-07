## Mini-JSTorch


Mini-JSTorch is a lightweight, `dependency-free` JavaScript neural network library designed for `education`, `experimentation`, and `small-scale models`.
It runs in Node.js and modern browsers, with a simple API inspired by PyTorch-style workflows.

This project prioritizes `clarity`, `numerical correctness`, and `accessibility` over performance or large-scale production use.

In this version `1.8.0`, we Introduce the **SoftmaxCrossEntropyLoss**, and **BCEWithLogitsLoss**

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
- ESM-first (`type: module`)
- Works in Node.js and browser environments
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
<body>
    <div id="output">
        <p>Status: <span id="status">Initializing...</span></p>
        <div id="training-log"></div>
        <div id="results" style="margin-top: 20px;"></div>
    </div>

    <script type="module">
        import { Sequential, Linear, ReLU, MSELoss, Adam, StepLR, Tanh } from 'https://unpkg.com/jstorch'; // DO NOT CHANGE
        
        const statusEl = document.getElementById('status');
        const trainingLogEl = document.getElementById('training-log');
        const resultsEl = document.getElementById('results');
        
        async function trainModel() {
            try {
                statusEl.textContent = 'Creating model...';
                
                const model = new Sequential([
                    new Linear(2, 16),
                    new Tanh(),
                    new Linear(16, 8),
                    new ReLU(),
                    new Linear(8, 1)
                ]);

                const X = [[0,0], [0,1], [1,0], [1,1]];
                const y = [[0], [1], [1], [0]];

                const criterion = new MSELoss();
                const optimizer = new Adam(model.parameters(), 0.1);
                const scheduler = new StepLR(optimizer, 25, 0.5);

                trainingLogEl.innerHTML = '<h4>Training Progress:</h4>';
                const logList = document.createElement('ul');
                trainingLogEl.appendChild(logList);

                statusEl.textContent = 'Training...';
                
                for (let epoch = 0; epoch < 1000; epoch++) {
                    const pred = model.forward(X);
                    const loss = criterion.forward(pred, y);
                    const grad = criterion.backward();
                    model.backward(grad);
                    optimizer.step();
                    scheduler.step();
                    
                    if (epoch % 100 === 0) {
                        const logItem = document.createElement('li');
                        logItem.textContent = `Epoch ${epoch}: Loss = ${loss.toFixed(6)}`;
                        logList.appendChild(logItem);
                        
                        // Update status every 100 epochs
                        statusEl.textContent = `Training... Epoch ${epoch}/1000 (Loss: ${loss.toFixed(6)})`;

                        await new Promise(resolve => setTimeout(resolve, 10));
                    }
                }

                statusEl.textContent = 'Training completed!';
                statusEl.style.color = 'green';

                resultsEl.innerHTML = '<h4>XOR Predictions:</h4>';
                const resultsTable = document.createElement('table');
                resultsTable.style.border = '1px solid #ccc';
                resultsTable.style.borderCollapse = 'collapse';
                resultsTable.style.width = '300px';
                
                // Table header
                const headerRow = document.createElement('tr');
                ['Input A', 'Input B', 'Prediction', 'Target'].forEach(text => {
                    const th = document.createElement('th');
                    th.textContent = text;
                    th.style.border = '1px solid #ccc';
                    th.style.padding = '8px';
                    headerRow.appendChild(th);
                });
                resultsTable.appendChild(headerRow);
                
                const predictions = model.forward(X);
                predictions.forEach((pred, i) => {
                    const row = document.createElement('tr');
                    
                    const cell1 = document.createElement('td');
                    cell1.textContent = X[i][0];
                    cell1.style.border = '1px solid #ccc';
                    cell1.style.padding = '8px';
                    cell1.style.textAlign = 'center';
                    
                    const cell2 = document.createElement('td');
                    cell2.textContent = X[i][1];
                    cell2.style.border = '1px solid #ccc';
                    cell2.style.padding = '8px';
                    cell2.style.textAlign = 'center';
                    
                    const cell3 = document.createElement('td');
                    cell3.textContent = pred[0].toFixed(4);
                    cell3.style.border = '1px solid #ccc';
                    cell3.style.padding = '8px';
                    cell3.style.textAlign = 'center';
                    cell3.style.fontWeight = 'bold';
                    cell3.style.color = Math.abs(pred[0] - y[i][0]) < 0.1 ? 'green' : 'red';
                    
                    const cell4 = document.createElement('td');
                    cell4.textContent = y[i][0];
                    cell4.style.border = '1px solid #ccc';
                    cell4.style.padding = '8px';
                    cell4.style.textAlign = 'center';
                    
                    row.appendChild(cell1);
                    row.appendChild(cell2);
                    row.appendChild(cell3);
                    row.appendChild(cell4);
                    resultsTable.appendChild(row);
                });
                
                resultsEl.appendChild(resultsTable);

                const summary = document.createElement('div');
                summary.style.marginTop = '20px';
                summary.style.padding = '10px';
                summary.style.backgroundColor = '#f0f0f0';
                summary.style.borderRadius = '5px';
                summary.innerHTML = `
                    <p><strong>Model Architecture:</strong> 2 → 16 → 8 → 1</p>
                    <p><strong>Activation:</strong> Tanh → ReLU</p>
                    <p><strong>Loss Function:</strong> MSE</p>
                    <p><strong>Optimizer:</strong> Adam (LR: 0.1)</p>
                    <p><strong>Epochs:</strong> 1000</p>
                `;
                resultsEl.appendChild(summary);
                
            } catch (error) {
                statusEl.textContent = `Error: ${error.message}`;
                statusEl.style.color = 'red';
                console.error(error);
            }
        }

        trainModel();
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

```bash
npm install mini-jstorch
```
Node.js v18+ or any modern browser with ES module support is recommended.

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
