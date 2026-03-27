## Mini-JSTorch (MAJOR UPDATE)

---

Mini-JSTorch is a lightweight, `dependency-free` JavaScript neural network library designed for `education`, `experimentation`, and `small-scale models`.
It runs in Node.js and modern browsers, with a simple API inspired by PyTorch-style workflows.

This project prioritizes `clarity`, `numerical correctness`, and `accessibility` over performance or large-scale production use.

In this version `2.0.0`, we introduce:
- **Fixed Linear layer cache** (critical bug fix for training)
- **Fixed GELU gradient calculation**
- **Fixed MSELoss gradient scaling**
- **Optimized Softmax gradient** (O(n²) → O(n))
- **Improved Tokenizer** with proper PAD/UNK separation
- **Added Sequential.zeroGrad(), train(), eval(), stateDict() methods**

---

**⚠️ BREAKING CHANGES in v2.0.0:**
- Tokenizer API: `tokenizeBatch()` → `transform()`, `detokenizeBatch()` → `inverseTransform()`
- Tokenizer now uses `<PAD>` at index 0 and `<UNK>` at index 1
- MSELoss gradient scale now matches PyTorch behavior

---

# Overview

**Mini-JSTorch provides a minimal neural network engine implemented entirely in plain JavaScript.**

*It is intended for:*

- learning how neural networks work internally
- experimenting with small models
- running simple training loops in the browser
- environments where large frameworks are unnecessary or unavailable

`mini-jstorch is intentionally designed to be small, readable, and easy to debug.`

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
- CrossEntropyLoss (*legacy*, use **SoftmaxCrossEntropy** instead)
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
- Dropout
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

## Multi-class Classification (SoftmaxCrossEntropy)

```javascript
import {
  Sequential,
  Linear,
  ReLU,
  SoftmaxCrossEntropyLoss,
  Adam
} from "./src/jstorch.js";

const model = new Sequential([
  new Linear(2, 8),
  new ReLU(),
  new Linear(8, 2) // logits output
]);

const X = [
  [0,0], [0,1], [1,0], [1,1]
];

const Y = [
  [1,0], [0,1], [0,1], [1,0]
];

const lossFn = new SoftmaxCrossEntropyLoss();
const optimizer = new Adam(model.parameters(), {lr: 0.1});

for (let epoch = 1; epoch <= 300; epoch++) {
  const logits = model.forward(X);
  const loss = lossFn.forward(logits, Y);
  const grad = lossFn.backward();
  model.backward(grad);
  optimizer.step();
  
  // Zero gradients for next iteration
  model.zeroGrad();

  if (epoch % 50 === 0) {
    console.log(`Epoch ${epoch}, Loss: ${loss.toFixed(6)}`);
  }
}
```
`Important:` Do not combine `SoftmaxCrossEntropyLoss` with a `Softmax` layer.

## Binary Classifiaction (BCEWithLogitsLoss)

```javascript
import {
  Sequential,
  Linear,
  ReLU,
  BCEWithLogitsLoss,
  Adam
} from "./src/jstorch.js";

const model = new Sequential([
  new Linear(2, 8),
  new ReLU(),
  new Linear(8, 1) // logit output
]);

const X = [
  [0,0], [0,1], [1,0], [1,1]
];
const Y = [
  [0], [1], [1], [0]
];

const lossFn = new BCEWithLogitsLoss();
const optimizer = new Adam(model.parameters(), {lr: 0.1});

for (let epoch = 1; epoch <= 300; epoch++) {
  const logits = model.forward(X);
  const loss = lossFn.forward(logits, Y);
  const grad = lossFn.backward();
  model.backward(grad);
  optimizer.step();
  model.zeroGrad();
  
  // Print progress every 50 epochs
  if (epoch % 50 === 0) {
    const probs = logits.map(p => 1 / (1 + Math.exp(-p[0])));
    console.log(`Epoch ${epoch} | Loss: ${loss.toFixed(6)}`);
    probs.forEach((prob, i) => {
      const pred = prob > 0.5 ? 1 : 0;
      console.log(`  [${X[i]}] → prob: ${prob.toFixed(4)} (${pred}) | target: ${Y[i][0]}`);
    });
    console.log('');
  }
}

// Final evaluation
console.log("\nTraining Complete\n");
model.eval(); 

const finalLogits = model.forward(X);
const finalProbs = finalLogits.map(p => 1 / (1 + Math.exp(-p[0])));

console.log("Final Results:");
let correct = 0;
finalProbs.forEach((prob, i) => {
  const pred = prob > 0.5 ? 1 : 0;
  const target = Y[i][0];
  const isCorrect = pred === target;
  if (isCorrect) correct++;
  console.log(`  [${X[i]}] → ${prob.toFixed(4)} (${pred}) | target: ${target} ${isCorrect ? '✓' : '✗'}`);
});
console.log(`\nAccuracy: ${(correct / X.length * 100).toFixed(2)}%`);
```
`Important:` Do not combine `BCEWithLogitsLoss` with a `Sigmoid` layer.

---

# Save & Load Models 

```javascript
// WARN: Error/Bug may be expected for this time!
import { saveModel, loadModel, Sequential } from "./src/jstorch.js";

const json = saveModel(model);
const model2 = new Sequential([...]); // same architecture
loadModel(model2, json);
```

---

# Demos

See the `demo/` directory for runnable examples!
- `demo/fu_fun.js`
- `demo/MakeModel.js`
- `demo/scheduler.js`
- `demo/xor_classification.js`
- `demo/linear_regression.js`


```bash
node demo/<fileNameInDemo>.js
```
**Make sure your directory while run this at root folder!**

---

# Design Notes & Limitations 

- Training logic is 2D-first: `[batch][features]`
- Higher-dimensional data is reshaped internally by specific layers (e.g. Conv2D, Flatten)
- No automatic broadcasting or autograd graph
- Some components (Conv2D, BatchNorm2D, Dropout) are educational / experimental
- Not intended for large-scale or production ML workloads

---

# License

MIT License

Copyright (c) 2024
rizal-editors

---
