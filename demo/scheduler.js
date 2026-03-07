// Example: Test ALL learning rate schedulers in mini-jstorch with mini-jstorch optimizers

import { SGD, StepLR, LambdaLR, ReduceLROnPlateau, Tensor } from "../src/jstorch.js";

const param = { param: [[1, 2], [3, 4]], grad: [[0, 0], [0, 0]] };
const optimizer = new SGD([param], 0.1);

// --- Test StepLR ---
console.log("Testing StepLR...");
const stepScheduler = new StepLR(optimizer, 3, 0.5);
for (let epoch = 1; epoch <= 10; epoch++) {
    stepScheduler.step();
    console.log(`Epoch ${epoch}: LR = ${optimizer.lr.toFixed(4)}`);
}

// --- Test LambdaLR ---
console.log("\nTesting LambdaLR...");
optimizer.lr = 0.1; // Reset LR
const lambdaScheduler = new LambdaLR(optimizer, epoch => 1.0 / (1 + epoch));
for (let epoch = 1; epoch <= 5; epoch++) {
    lambdaScheduler.step();
    console.log(`Epoch ${epoch}: LR = ${optimizer.lr.toFixed(4)}`);
}

// --- Test ReduceLROnPlateau ---
console.log("\nTesting ReduceLROnPlateau...");
optimizer.lr = 0.1; // Reset LR
const plateauScheduler = new ReduceLROnPlateau(optimizer, {
    patience: 2,
    factor: 0.5,
    min_lr: 0.01,
    verbose: true
});

// Simulate training with plateauing loss
const losses = [0.9, 0.8, 0.7, 0.69, 0.68, 0.68, 0.68, 0.67, 0.67, 0.67];
console.log("Simulated training with plateauing loss:");
for (let epoch = 0; epoch < losses.length; epoch++) {
    plateauScheduler.step(losses[epoch]);
    console.log(`Epoch ${epoch + 1}: Loss = ${losses[epoch].toFixed(3)}, LR = ${optimizer.lr.toFixed(4)}, Wait = ${plateauScheduler.wait}`);
}

// --- Test ReduceLROnPlateau with Cooldown ---
console.log("\nTesting ReduceLROnPlateau with Cooldown...");
optimizer.lr = 0.1; // Reset LR
const plateauWithCooldown = new ReduceLROnPlateau(optimizer, {
    patience: 2,
    factor: 0.5,
    min_lr: 0.01,
    cooldown: 2,
    verbose: true
});

// Simulate training with multiple plateaus
const losses2 = [0.9, 0.9, 0.9, 0.9, 0.8, 0.8, 0.8, 0.8, 0.7, 0.7];
console.log("Simulated training with cooldown:");
for (let epoch = 0; epoch < losses2.length; epoch++) {
    plateauWithCooldown.step(losses2[epoch]);
    console.log(`Epoch ${epoch + 1}: Loss = ${losses2[epoch].toFixed(3)}, LR = ${optimizer.lr.toFixed(4)}, Wait = ${plateauWithCooldown.wait}, Cooldown = ${plateauWithCooldown.cooldown_counter}`);
}

// --- Summary ---
console.log("\nSCHEDULER SUMMARY:");
console.log(`StepLR: ${stepScheduler.last_epoch} epochs processed`);
console.log(`LambdaLR: ${lambdaScheduler.last_epoch} epochs processed`);
console.log(`ReduceLROnPlateau: ${plateauScheduler.num_reductions} LR reductions`);
console.log(`ReduceLROnPlateau with Cooldown: ${plateauWithCooldown.num_reductions} LR reductions`);

console.log("\nAll schedulers tested successfully!");