import { Sequential, Linear, ReLU, BCEWithLogitsLoss, Adam 
} from '../src/jstorch.js';

// Dataset XOR
const X = [[0,0], [0,1], [1,0], [1,1]];
const Y = [[0], [1], [1], [0]];

// Model
const model = new Sequential([
    new Linear(2, 8),
    new ReLU(),
    new Linear(8, 1)
]);

const lossFn = new BCEWithLogitsLoss();
const optimizer = new Adam(model.parameters(), {lr: 0.1});

console.log("=== XOR Binary Classification ===\n");

for (let epoch = 1; epoch <= 200; epoch++) {
    const logits = model.forward(X);
    const loss = lossFn.forward(logits, Y);
    const grad = lossFn.backward();
    model.backward(grad);
    optimizer.step();
    model.zeroGrad();
    
    if (epoch % 50 === 0) {
        const probs = logits.map(p => 1 / (1 + Math.exp(-p[0])));
        console.log(`Epoch ${epoch} | Loss: ${loss.toFixed(6)}`);
        probs.forEach((p, i) => console.log(`  [${X[i]}] → ${p.toFixed(4)} (target: ${Y[i][0]})`));
        console.log('');
    }
}

// Evaluation
model.eval();
const finalLogits = model.forward(X);
const finalProbs = finalLogits.map(p => 1 / (1 + Math.exp(-p[0])));

let correct = 0;
finalProbs.forEach((p, i) => {
    const pred = p > 0.5 ? 1 : 0;
    if (pred === Y[i][0]) correct++;
    console.log(`[${X[i]}] → pred: ${pred} (${p.toFixed(4)}) | target: ${Y[i][0]} ${pred === Y[i][0] ? '✓' : '✗'}`);
});
console.log(`\nAccuracy: ${(correct / X.length * 100).toFixed(0)}%`);