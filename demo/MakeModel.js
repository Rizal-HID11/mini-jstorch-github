import { Sequential, Linear, ReLU, MSELoss, Adam, StepLR, Tanh } from '../src/jstorch.js';

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
const scheduler = new StepLR(optimizer, 25, 0.5); // LR * 0.5 every 25 epochs

console.log("Training Progress:");
for (let epoch = 0; epoch < 1000; epoch++) {
    const pred = model.forward(X);
    const loss = criterion.forward(pred, y);
    const grad = criterion.backward();
    model.backward(grad);
    optimizer.step();
    scheduler.step();
    
    if (epoch % 100 === 0) {
        console.log(`Epoch ${epoch}: Loss = ${loss.toFixed(6)}, LR = ${optimizer.lr.toFixed(6)}`);
    }
}

console.log("\nFinal Predictions:");
const predictions = model.forward(X);
predictions.forEach((pred, i) => {
    console.log(`Input: ${X[i]} -> ${pred[0].toFixed(4)} (target: ${y[i][0]})`);
});