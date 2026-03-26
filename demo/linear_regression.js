import { Sequential, Linear, MSELoss, Adam 
} from '../src/jstorch.js';

// Dataset: y = 2x
const X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]];
const Y = [[2], [4], [6], [8], [10], [12], [14], [16], [18], [20]];

const model = new Sequential([
    new Linear(1, 1)
]);

const lossFn = new MSELoss();
const optimizer = new Adam(model.parameters(), {lr: 0.1}); // ← LR 0.1

console.log("=== Linear Regression (y = 2x) ===\n");

for (let epoch = 1; epoch <= 500; epoch++) {
    const pred = model.forward(X);
    const loss = lossFn.forward(pred, Y);
    const grad = lossFn.backward();
    model.backward(grad);
    optimizer.step();
    model.zeroGrad();
    
    if (epoch % 100 === 0) {
        console.log(`Epoch ${epoch} | Loss: ${loss.toFixed(6)}`);
    }
}

model.eval();
const finalPred = model.forward(X);

console.log('\nResults:');
let totalError = 0;
finalPred.forEach((p, i) => {
    const error = Math.abs(p[0] - Y[i][0]);
    totalError += error;
    console.log(`  x=${X[i][0]} → pred: ${p[0].toFixed(2)} | actual: ${Y[i][0]} | error: ${error.toFixed(2)}`);
});
console.log(`\nAverage Error: ${(totalError / X.length).toFixed(2)}`);
console.log(`Weight (slope): ${model.layers[0].W[0][0].toFixed(4)} (expected: 2.0)`);
console.log(`Bias: ${model.layers[0].b[0].toFixed(4)} (expected: 0.0)`);