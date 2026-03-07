/*!
 * File: jstorch.js
 * Author: rizal-editors
 * License: MIT
 * Copyright (C) 2025 rizal-editors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

// --------------------------------------------------------------
// PLEASE READ THE README.md FILE BEFORE USING THIS PACKAGE!
// This package is designed to be used in a Node.js/browser environment.
// See the Documentation for more details.
// --------------------------------------------------------------

// ---------------------- DONOT USE THESE (ENGINE INTERNALS) ERROR/BUG ARE EXPECTED ----------------------
export function zeros(rows, cols) { 
    return Array.from({length:rows},()=>Array(cols).fill(0)); 
}

export function ones(rows, cols) { 
    return Array.from({length:rows},()=>Array(cols).fill(1)); 
}

export function randomMatrix(rows, cols, scale=null){
    // Auto-scale based on layer size (Xavier init)
    if (scale === null){
        scale = Math.sqrt(2.0 / (rows + cols));
    }

    return Array.from({length: rows}, () =>
        Array.from({length: cols}, () => (Math.random() * 2 - 1) * scale));
}

export function transpose(matrix){ 
    return matrix[0].map((_,i)=>matrix.map(row=>row[i])); 
}

export function addMatrices(a,b){ 
    return a.map((row,i)=>
        row.map((v,j)=>v+(b[i] && b[i][j]!==undefined?b[i][j]:0))
    ); 
}

export function dot(a,b){ 
    const res=zeros(a.length,b[0].length); 
    for(let i=0;i<a.length;i++) 
        for(let j=0;j<b[0].length;j++) 
            for(let k=0;k<a[0].length;k++) 
                res[i][j]+=a[i][k]*b[k][j]; 
    return res; 
}

export function softmax(x){ 
    const m=Math.max(...x); 
    const exps=x.map(v=>Math.exp(v-m)); 
    const s=exps.reduce((a,b)=>a+b,0); 
    return exps.map(v=>v/s); 
}

export function crossEntropy(pred,target){ 
    const eps=1e-12; 
    return -target.reduce((sum,t,i)=>sum+t*Math.log(pred[i]+eps),0); 
}

// ---------------------- USERS FRIENDLY UTILS (USE THIS FOR YOUR UTILS!) ----------------
export function fu_tensor(data, requiresGrad = false) {
    if (!Array.isArray(data) || !Array.isArray(data[0])) {
        throw new Error("fu_tensor: Data must be 2D array");
    }
    const tensor = new Tensor(data);
    tensor.requiresGrad = requiresGrad;
    return tensor;
}

// fu_add
export function fu_add(a, b) {
    if (!(a instanceof Tensor) && !(b instanceof Tensor)) {
        throw new Error("fu_add: At least one operand must be Tensor");
    }
    
    if (!(a instanceof Tensor)) {
        a = fu_tensor(Array(b.shape()[0]).fill().map(() => 
            Array(b.shape()[1]).fill(a)
        ));
    }
    
    if (!(b instanceof Tensor)) {
        b = fu_tensor(Array(a.shape()[0]).fill().map(() => 
            Array(a.shape()[1]).fill(b)
        ));
    }
    
    if (a.shape()[0] !== b.shape()[0] || a.shape()[1] !== b.shape()[1]) {
        throw new Error(`fu_add: Shape mismatch ${a.shape()} vs ${b.shape()}`);
    }
    
    return new Tensor(a.data.map((r, i) => r.map((v, j) => v + b.data[i][j])));
}

// fu_mul
export function fu_mul(a, b) {
    if (!(a instanceof Tensor) && !(b instanceof Tensor)) {
        throw new Error("fu_mul: At least one operand must be Tensor");
    }
    
    if (!(a instanceof Tensor)) {
        a = fu_tensor(Array(b.shape()[0]).fill().map(() => 
            Array(b.shape()[1]).fill(a)
        ));
    }
    
    if (!(b instanceof Tensor)) {
        b = fu_tensor(Array(a.shape()[0]).fill().map(() => 
            Array(a.shape()[1]).fill(b)
        ));
    }
    
    if (a.shape()[0] !== b.shape()[0] || a.shape()[1] !== b.shape()[1]) {
        throw new Error(`fu_mul: Shape mismatch ${a.shape()} vs ${b.shape()}`);
    }
    
    return new Tensor(a.data.map((r, i) => r.map((v, j) => v * b.data[i][j])));
}

// fu_matmul
export function fu_matmul(a, b) {
    if (!(a instanceof Tensor)) a = fu_tensor(a);
    if (!(b instanceof Tensor)) b = fu_tensor(b);
    
    if (a.shape()[1] !== b.shape()[0]) {
        throw new Error(`fu_matmul: Inner dimension mismatch ${a.shape()[1]} vs ${b.shape()[0]}`);
    }
    
    return new Tensor(dot(a.data, b.data));
}

// fu_sum
export function fu_sum(tensor) {
    if (!(tensor instanceof Tensor)) tensor = fu_tensor(tensor);
    const total = tensor.data.flat().reduce((a, b) => a + b, 0);
    return new Tensor([[total]]);
}

// fu_mean
export function fu_mean(tensor) {
    if (!(tensor instanceof Tensor)) tensor = fu_tensor(tensor);
    const totalElements = tensor.shape()[0] * tensor.shape()[1];
    const sum = fu_sum(tensor).data[0][0];
    return new Tensor([[sum / totalElements]]);
}

// fu_relu
export function fu_relu(tensor) {
    if (!(tensor instanceof Tensor)) tensor = fu_tensor(tensor);
    return new Tensor(tensor.data.map(r => r.map(v => Math.max(0, v))));
}

// fu_sigmoid
export function fu_sigmoid(tensor) {
    if (!(tensor instanceof Tensor)) tensor = fu_tensor(tensor);
    const fn = v => 1 / (1 + Math.exp(-v));
    return new Tensor(tensor.data.map(r => r.map(fn)));
}

// fu_tanh
export function fu_tanh(tensor) {
    if (!(tensor instanceof Tensor)) tensor = fu_tensor(tensor);
    return new Tensor(tensor.data.map(r => r.map(v => Math.tanh(v))));
}

// fu_softmax
export function fu_softmax(tensor) {
    if (!(tensor instanceof Tensor)) tensor = fu_tensor(tensor);
    const result = tensor.data.map(row => {
        const maxVal = Math.max(...row);
        const exps = row.map(v => Math.exp(v - maxVal));
        const sumExps = exps.reduce((a, b) => a + b, 0);
        return exps.map(v => v / sumExps);
    });
    return new Tensor(result);
}

// fu_flatten - Flatten tensor to 1D
export function fu_flatten(tensor) {
    if (!(tensor instanceof Tensor)) tensor = fu_tensor(tensor);
    return new Tensor([tensor.data.flat()]);
}

// fu_reshape
export function fu_reshape(tensor, rows, cols) {
    if (!(tensor instanceof Tensor)) tensor = fu_tensor(tensor);
    const flat = tensor.data.flat();
    if (flat.length !== rows * cols) {
        throw new Error(`fu_reshape: Size mismatch ${flat.length} vs ${rows * cols}`);
    }
    
    const result = [];
    for (let i = 0; i < rows; i++) {
        result.push(flat.slice(i * cols, i * cols + cols));
    }
    return new Tensor(result);
}

// fu_stack
export function fu_stack(tensors) {
    if (!tensors.every(t => t instanceof Tensor)) {
        throw new Error("fu_stack: All inputs must be Tensors");
    }
    
    const firstShape = tensors[0].shape();
    if (!tensors.every(t => t.shape()[0] === firstShape[0] && t.shape()[1] === firstShape[1])) {
        throw new Error("fu_stack: All tensors must have same shape");
    }
    
    const stacked = tensors.map(t => t.data);
    return new Tensor(stacked);
}

// ---------------------- Tensor ----------------------
export class Tensor {
    constructor(data){ this.data=data; this.grad=zeros(data.length,data[0].length); }
    shape(){ return [this.data.length,this.data[0].length]; }
    add(t){ return t instanceof Tensor?this.data.map((r,i)=>r.map((v,j)=>v+t.data[i][j])):this.data.map(r=>r.map(v=>v+t)); }
    sub(t){ return t instanceof Tensor?this.data.map((r,i)=>r.map((v,j)=>v-t.data[i][j])):this.data.map(r=>r.map(v=>v-t)); }
    mul(t){ return t instanceof Tensor?this.data.map((r,i)=>r.map((v,j)=>v*t.data[i][j])):this.data.map(r=>r.map(v=>v*t)); }
    matmul(t){ if(t instanceof Tensor) return dot(this.data,t.data); else throw new Error("matmul requires Tensor"); }
    transpose(){ return transpose(this.data); }
    flatten(){ return this.data.flat(); }
    static zeros(r,c){ return new Tensor(zeros(r,c)); }
    static ones(r,c){ return new Tensor(ones(r,c)); }
    static random(r,c,scale=0.1){ return new Tensor(randomMatrix(r,c,scale)); }
}

// ---------------------- Layers ----------------------
export class Linear {
    constructor(inputDim, outputDim){
        this.W = randomMatrix(inputDim, outputDim);
        this.b = Array(outputDim).fill(0);
        this.gradW = zeros(inputDim, outputDim);
        this.gradb = Array(outputDim).fill(0);
        this.x = null;
        this.originalShape = null; // Track input shape
    }

    forward(x){
        // Handle both [batch, features] and [batch, 1, features]
        this.originalShape = this._getShapeType(x);
        
        if (this.originalShape === '3d') {
            // Convert from [batch, 1, features] to [batch, features]
            this.x = x.map(sample => sample[0]);
        } else {
            // Already in [batch, features] format
            this.x = x;
        }
        
        const out = dot(this.x, this.W);
        return out.map((row, i) => row.map((v, j) => v + this.b[j]));
    }

    backward(grad){
        // Compute gradients
        for(let i = 0; i < this.W.length; i++) {
            for(let j = 0; j < this.W[0].length; j++) {
                this.gradW[i][j] = this.x.reduce((sum, row, k) => sum + row[i] * grad[k][j], 0);
            }
        }
        
        for(let j = 0; j < this.b.length; j++) {
            this.gradb[j] = grad.reduce((sum, row) => sum + row[j], 0);
        }

        const gradInput = zeros(this.x.length, this.W.length);
        for(let i = 0; i < this.x.length; i++) {
            for(let j = 0; j < this.W.length; j++) {
                for(let k = 0; k < this.W[0].length; k++) {
                    gradInput[i][j] += grad[i][k] * this.W[j][k];
                }
            }
        }
        
        //Convert back to original shape if needed
        if (this.originalShape === '3d') {
            return gradInput.map(row => [row]); // Back to [batch, 1, features]
        }
        return gradInput;
    }

    _getShapeType(x) {
        if (Array.isArray(x[0]) && Array.isArray(x[0][0]) && !Array.isArray(x[0][0][0])) {
            return '3d'; // [batch, 1, features]
        } else if (Array.isArray(x[0]) && !Array.isArray(x[0][0])) {
            return '2d'; // [batch, features]  
        } else {
            throw new Error(`Unsupported input shape for Linear layer`);
        }
    }

    parameters(){ 
        return [ 
            {param: this.W, grad: this.gradW}, 
            {param: [this.b], grad: [this.gradb]} 
        ]; 
    }
}

export class Flatten {
    constructor() {
        this.originalShape = null;
    }
    
    forward(x) {
        // Always convert to [batch, features] format
        this.originalShape = x.map(sample => this._getShape(sample));
        
        return x.map(sample => {
            const flat = this._flatten(sample);
            return flat; // Return as 1D array for [batch, features] compatibility
        });
    }
    
    backward(grad) {
        // grad is [batch, features], reshape back to original shape
        return grad.map((flatGrad, batchIdx) => {
            const shape = this.originalShape[batchIdx];
            return this._unflatten(flatGrad, shape);
        });
    }
    
    _getShape(sample) {
        if (Array.isArray(sample[0]) && Array.isArray(sample[0][0])) {
            return {
                type: '3d',
                dims: [sample.length, sample[0].length, sample[0][0].length]
            };
        } else if (Array.isArray(sample[0])) {
            return {
                type: '2d', 
                dims: [sample.length, sample[0].length]
            };
        } else {
            return {
                type: '1d',
                dims: [sample.length]
            };
        }
    }
    
    _flatten(sample) {
        if (Array.isArray(sample[0]) && Array.isArray(sample[0][0])) {
            return sample.flat(2); // [channels, height, width] -> flat
        } else if (Array.isArray(sample[0])) {
            return sample.flat(); // [height, width] -> flat
        } else {
            return sample; // already flat
        }
    }
    
    _unflatten(flat, shape) {
        if (shape.type === '3d') {
            const [channels, height, width] = shape.dims;
            const result = [];
            let index = 0;
            for (let c = 0; c < channels; c++) {
                const channel = [];
                for (let h = 0; h < height; h++) {
                    const row = [];
                    for (let w = 0; w < width; w++) {
                        row.push(flat[index++]);
                    }
                    channel.push(row);
                }
                result.push(channel);
            }
            return result;
        } else if (shape.type === '2d') {
            const [height, width] = shape.dims;
            const result = [];
            for (let h = 0; h < height; h++) {
                result.push(flat.slice(h * width, h * width + width));
            }
            return result;
        } else {
            return flat; // 1d
        }
    }
    
    parameters() { return []; }
}

// ---------------------- Conv2D ----------------------
export class Conv2D {
    constructor(inC, outC, kernel, stride=1, padding=0){
        this.inC = inC; 
        this.outC = outC; 
        this.kernel = kernel;
        this.stride = stride; 
        this.padding = padding;
        this.W = Array(outC).fill().map(() => 
            Array(inC).fill().map(() => randomMatrix(kernel, kernel))
        );
        this.gradW = Array(outC).fill().map(() => 
            Array(inC).fill().map(() => zeros(kernel, kernel))
        );
        this.x = null;
    }

    pad2D(input, pad){
        // Input is single channel [height, width]
        if (!input || !input.length) return input;
        
        const rows = input.length + 2 * pad;
        const cols = input[0].length + 2 * pad;
        const out = Array.from({length: rows}, () => Array(cols).fill(0));
        
        for(let i = 0; i < input.length; i++) {
            for(let j = 0; j < input[0].length; j++) {
                out[i + pad][j + pad] = input[i][j];
            }
        }
        return out;
    }

    conv2DSingle(input, kernel) {
        const rows = Math.floor((input.length - kernel.length) / this.stride) + 1;
        const cols = Math.floor((input[0].length - kernel[0].length) / this.stride) + 1;
        const out = zeros(rows, cols);
        
        for(let i = 0; i < rows; i++) {
            for(let j = 0; j < cols; j++) {
                let sum = 0;
                for(let ki = 0; ki < kernel.length; ki++) {
                    for(let kj = 0; kj < kernel[0].length; kj++) {
                        const inputRow = i * this.stride + ki;
                        const inputCol = j * this.stride + kj;
                        sum += input[inputRow][inputCol] * kernel[ki][kj];
                    }
                }
                out[i][j] = sum;
            }
        }
        return out;
    }

    forward(batch) {
        this.x = batch;
        return batch.map(sample => {
            const channelsOut = [];
            for(let oc = 0; oc < this.outC; oc++) {
                let outChan = null;
                for(let ic = 0; ic < this.inC; ic++) {
                    let inputChan = sample[ic];
                    if(this.padding > 0) {
                        inputChan = this.pad2D(inputChan, this.padding);
                    }
                    
                    const conv = this.conv2DSingle(inputChan, this.W[oc][ic]);
                    
                    if(outChan === null) {
                        outChan = conv;
                    } else {
                        outChan = addMatrices(outChan, conv);
                    }
                }
                channelsOut.push(outChan);
            }
            return channelsOut;
        });
    }

    backward(grad) {
        const batchSize = this.x.length;
        const gradW = this.W.map(oc => oc.map(ic => zeros(this.kernel, this.kernel)));
        const gradInput = this.x.map(sample => 
            sample.map(chan => zeros(chan.length, chan[0].length))
        );

        for (let b = 0; b < batchSize; b++) {
            for (let oc = 0; oc < this.outC; oc++) {
                for (let ic = 0; ic < this.inC; ic++) {
                    const outGrad = grad[b][oc];
                    
                    // Compute gradW
                    for (let i = 0; i < this.kernel; i++) {
                        for (let j = 0; j < this.kernel; j++) {
                            let sum = 0;
                            for (let y = 0; y < outGrad.length; y++) {
                                for (let x = 0; x < outGrad[0].length; x++) {
                                    const inY = y * this.stride + i;
                                    const inX = x * this.stride + j;
                                    if (inY < this.x[b][ic].length && inX < this.x[b][ic][0].length) {
                                        sum += this.x[b][ic][inY][inX] * outGrad[y][x];
                                    }
                                }
                            }
                            gradW[oc][ic][i][j] += sum;
                        }
                    }

                    // Compute gradInput
                    for (let y = 0; y < outGrad.length; y++) {
                        for (let x = 0; x < outGrad[0].length; x++) {
                            for (let ki = 0; ki < this.kernel; ki++) {
                                for (let kj = 0; kj < this.kernel; kj++) {
                                    const inY = y * this.stride + ki;
                                    const inX = x * this.stride + kj;
                                    if (inY < gradInput[b][ic].length && inX < gradInput[b][ic][0].length) {
                                        gradInput[b][ic][inY][inX] += 
                                            this.W[oc][ic][ki][kj] * outGrad[y][x];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        this.gradW = gradW;
        return gradInput;
    }

    parameters() { 
        return this.W.flatMap((w, oc) => 
            w.map((wc, ic) => ({
                param: wc, 
                grad: this.gradW[oc][ic]
            }))
        ); 
    }
}

// ---------------------- Sequential ----------------------
export class Sequential {
    constructor(layers=[]){ this.layers=layers; }
    forward(x){ return this.layers.reduce((acc,l)=>l.forward(acc), x); }
    backward(grad){ return this.layers.reduceRight((g,l)=>l.backward(g), grad); }
    parameters(){ return this.layers.flatMap(l=>l.parameters?l.parameters():[]); }
}

// ---------------------- Activations ----------------------
export class ReLU{ 
    constructor(){ this.mask = null; this.originalShape = null; } 
    
    forward(x){ 
        this.originalShape = this._getShapeType(x);
        
        if (this.originalShape === '3d') {
            // Handle [batch, 1, features]
            this.mask = x.map(sample => sample[0].map(v => v > 0));
            return x.map(sample => [sample[0].map(v => Math.max(0, v))]);
        } else {
            // Handle [batch, features]
            this.mask = x.map(row => row.map(v => v > 0));
            return x.map(row => row.map(v => Math.max(0, v)));
        }
    } 
    
    backward(grad){ 
        if (this.originalShape === '3d') {
            return grad.map((sample, i) => 
                [sample[0].map((v, j) => this.mask[i][j] ? v : 0)]
            );
        } else {
            return grad.map((row, i) => 
                row.map((v, j) => this.mask[i][j] ? v : 0)
            );
        }
    }
    
    _getShapeType(x) {
        if (Array.isArray(x[0]) && Array.isArray(x[0][0]) && !Array.isArray(x[0][0][0])) {
            return '3d';
        } else if (Array.isArray(x[0]) && !Array.isArray(x[0][0])) {
            return '2d';
        } else {
            throw new Error(`Unsupported input shape for ReLU`);
        }
    }
}

// ---------------------- Softmax ----------------------
export class Softmax {
    constructor(dim = -1) {
        this.dim = dim;
        this.output = null;
        this.input = null;
    }

    forward(x) {
        this.input = x;
        
        // x: [batch_size, num_classes]
        this.output = x.map(row => {
            const maxVal = Math.max(...row);
            const exps = row.map(v => Math.exp(v - maxVal));
            const sumExps = exps.reduce((a, b) => a + b, 0);
            return exps.map(v => v / sumExps);
        });
        return this.output;
    }

    backward(grad) {
        // grad: [batch_size, num_classes] - gradient from next layer
        const batchSize = grad.length;
        const numClasses = grad[0].length;
        
        const gradInput = zeros(batchSize, numClasses);
        
        for (let i = 0; i < batchSize; i++) {
            const s = this.output[i]; // Softmax output for this sample
            const gradOut = grad[i];  // Gradient from loss
            
            // Compute Jacobian matrix: J_ij = s_i * (δ_ij - s_j)
            for (let j = 0; j < numClasses; j++) {
                let sum = 0;
                for (let k = 0; k < numClasses; k++) {
                    // J[j][k] = s[j] * ((j === k ? 1 : 0) - s[k])
                    const jacobian = s[j] * ((j === k ? 1 : 0) - s[k]);
                    sum += jacobian * gradOut[k];
                }
                gradInput[i][j] = sum;
            }
        }
        
        return gradInput;
    }

    parameters() {
        return []; // Softmax has no trainable parameters
    }
}

// ---------------------- Tokenizer ----------------------
export class Tokenizer {
    constructor(vocabSize = 2000){
        this.vocabSize = vocabSize;
        this.wordToIndex = new Map();
        this.indexToWord = new Map();
        this.fitted = false;
    }

    fit(texts){
        const wordCounts = new Map();

        // Count word frequencies from all texts
        texts.forEach(text => {
            const words = this._preprocess(text);
            words.forEach(word => {
                wordCounts.set(word, (wordCounts.get(word) || 0) + 1);
            });
        });

        // Sort by frequency and take top words
        const sortedWords = [...wordCounts.entries()]
            .sort((a, b) => a[1] - a[1])
            .slice(0, this.vocabSize - 1); // Reverse 1 for unknown

        // Build vocabulary
        this.wordToIndex.clear();
        this.indexToWord.clear();

        // Add unk token
        this.wordToIndex.set('<UNK>', 0);
        this.indexToWord.set(0, '<UNK>');

        // Add most frequent words
        sortedWords.forEach(([word], index) =>{
            this.wordToIndex.set(word, index + 1);
            this.indexToWord.set(index + 1, word);
        })

        this.fitted = true;
        return this;
    }

    tokenize(text){
        if (!this.fitted) throw new Error("Tokenizer not fitted. Call fit() first.");

        const words = this._preprocess(text);
        return words.map(word => this.wordToIndex.get(word) || 0);
    }

    tokenizeBatch(texts, maxLength=null){
        if (!this.fitted) throw new Error("Tokenizer not fitted. Call fit() first.");

        return texts.map(text => {
            const tokens = this.tokenize(text);

            if (maxLength !== null){
                // Pad or truncate to maxLength
                if (tokens.length > maxLength){
                    return tokens.slice(0, maxLength);
                } else {
                    return [...tokens, ...Array(maxLength - tokens.length).fill(0)];
                }
            }

            return tokens;
        })
    }

    detokenize(tokens){
        return tokens.map(token => this.indexToWord.get(token) || '<UNK>').join(' ');
    }

    detokenizeBatch(tokenBatches){
        return tokenBatches.map(tokens => this.detokenize(tokens));
    }

    getVocabSize(){
        return this.wordToIndex.size;
    }

    _preprocess(text) {
        return text.toLowerCase()
                  .replace(/[^\w\s]/g, ' ')  // Remove punctuation
                  .split(/\s+/)             // Split by whitespace
                  .filter(word => word.length > 0); // Remove empty strings
    }
}

// I'm too lazy to break lines here, so everything stays in one line
export class Sigmoid{ constructor(){ this.out=null; } forward(x){ const fn=v=>1/(1+Math.exp(-v)); this.out=x.map(r=>r.map(fn)); return this.out; } backward(grad){ return grad.map((r,i)=>r.map((v,j)=>v*this.out[i][j]*(1-this.out[i][j]))); } }
export class Tanh{ constructor(){ this.out=null; } forward(x){ this.out=x.map(r=>r.map(v=>Math.tanh(v))); return this.out; } backward(grad){ return grad.map((r,i)=>r.map((v,j)=>v*(1-this.out[i][j]**2))); } }
export class LeakyReLU{ constructor(alpha=0.01){ this.alpha=alpha; this.out=null; } forward(x){ this.out=x.map(r=>r.map(v=>v>0?v:v*this.alpha)); return this.out; } backward(grad){ return grad.map((r,i)=>r.map((v,j)=>v*(this.out[i][j]>0?1:this.alpha))); } }
export class GELU{ constructor(){ this.out=null; } forward(x){ const fn=v=>0.5*v*(1+Math.tanh(Math.sqrt(2/Math.PI)*(v+0.044715*v**3))); this.out=x.map(r=>r.map(fn)); return this.out; } backward(grad){ return grad.map((r,i)=>r.map(v=>v*1)); } }

// ---------------------- Dropout ----------------------
export class Dropout{ constructor(p=0.5){ this.p=p; } forward(x){ return x.map(r=>r.map(v=>v*Math.random()>=this.p?v:0)); } backward(grad){ return grad.map(r=>r.map(v=>v*(1-this.p))); } }

// ---------------------- Losses ----------------------
export class MSELoss{ forward(pred,target){ this.pred=pred; this.target=target; const losses=pred.map((row,i)=>row.reduce((sum,v,j)=>sum+(v-target[i][j])**2,0)/row.length); return losses.reduce((a,b)=>a+b,0)/pred.length; } backward(){ return this.pred.map((row,i)=>row.map((v,j)=>2*(v-this.target[i][j])/row.length)); } }
export class CrossEntropyLoss{ forward(pred,target){ this.pred=pred; this.target=target; const losses=pred.map((p,i)=>crossEntropy(softmax(p),target[i])); return losses.reduce((a,b)=>a+b,0)/pred.length; } backward(){ return this.pred.map((p,i)=>{ const s=softmax(p); return s.map((v,j)=>(v-this.target[i][j])/this.pred.length); }); } }

export class SoftmaxCrossEntropyLoss {
  forward(logits, targets) {
    this.targets = targets;
    const batch = logits.length;

    // stable softmax
    this.probs = logits.map(row => {
      const max = Math.max(...row);
      const exps = row.map(v => Math.exp(v - max));
      const sum = exps.reduce((a,b)=>a+b, 0);
      return exps.map(v => v / sum);
    });

    let loss = 0;
    for (let i = 0; i < batch; i++) {
      for (let j = 0; j < this.probs[i].length; j++) {
        if (targets[i][j] === 1) {
          loss -= Math.log(this.probs[i][j] + 1e-12);
        }
      }
    }

    return loss / batch;
  }

  backward() {
    const batch = this.targets.length;
    return this.probs.map((row,i) =>
      row.map((p,j) => (p - this.targets[i][j]) / batch)
    );
  }
}

export class BCEWithLogitsLoss {
  forward(logits, targets) {
    this.logits = logits;
    this.targets = targets;
    const batch = logits.length;
    let loss = 0;

    for (let i = 0; i < batch; i++) {
      for (let j = 0; j < logits[i].length; j++) {
        const x = logits[i][j];
        const y = targets[i][j];
        // stable BCE
        loss += Math.max(x, 0) - x*y + Math.log(1 + Math.exp(-Math.abs(x)));
      }
    }

    return loss / batch;
  }

  backward() {
    const batch = this.logits.length;
    return this.logits.map((row,i) =>
      row.map((x,j) => {
        const sigmoid = 1 / (1 + Math.exp(-x));
        return (sigmoid - this.targets[i][j]) / batch;
      })
    );
  }
}

// ---------------------- Optimizers ----------------------
export class Adam{
    constructor(params, lr = 0.001, b1 = 0.9, b2 = 0.999, eps = 1e-8, max_grad_norm = 1.0){
        // Handle both parameter styles: (params, lr) OR (params, {lr, ...})
        if (typeof lr === 'object') {
            // Options object provided
            const options = lr;
            this.lr = options.lr || 0.001;
            this.beta1 = options.b1 || options.beta1 || 0.9;
            this.beta2 = options.b2 || options.beta2 || 0.999;
            this.eps = options.eps || 1e-8;
            this.max_grad_norm = options.max_grad_norm || 1.0;
        } else {
            // Individual parameters provided
            this.lr = lr;
            this.beta1 = b1;
            this.beta2 = b2;
            this.eps = eps;
            this.max_grad_norm = max_grad_norm;
        }
        
        this.params = params;
        this.m = params.map(p => zeros(p.param.length, p.param[0].length || 1));
        this.v = params.map(p => zeros(p.param.length, p.param[0].length || 1));
        this.t = 0;
        
    }

    step(){
        this.t++;
        this.params.forEach((p, idx) => {
            // Calculate gradient norm for clipping
            let grad_norm_sq = 0;
            for (let i = 0; i < p.param.length; i++){
                for (let j = 0; j < (p.param[0].length || 1); j++){
                    const grad_val = p.grad[i] && p.grad[i][j] !== undefined ? p.grad[i][j] : 0;
                    grad_norm_sq += grad_val * grad_val;
                }
            }

            const grad_norm = Math.sqrt(grad_norm_sq);
            const clip_scale = grad_norm > this.max_grad_norm ? this.max_grad_norm / grad_norm : 1.0;

            // Update with clipped gradients
            for (let i = 0; i < p.param.length; i++){
                for(let j = 0; j < (p.param[0].length || 1); j++){
                    if (p.grad[i] && p.grad[i][j] !== undefined){
                        const g = p.grad[i][j] * clip_scale;
                        this.m[idx][i][j] = this.beta1 * this.m[idx][i][j] + (1 - this.beta1) * g;
                        this.v[idx][i][j] = this.beta2 * this.v[idx][i][j] + (1 - this.beta2) * g * g;
                        const mHat = this.m[idx][i][j] / (1 - Math.pow(this.beta1, this.t));
                        const vHat = this.v[idx][i][j] / (1 - Math.pow(this.beta2, this.t));
                        p.param[i][j] -= this.lr * mHat / (Math.sqrt(vHat) + this.eps);
                    }
                }
            }
        });
    }
}

// ---------------------- AdamW Optimizer ----------------------
export class AdamW {
    constructor(params, options = {}) {
        const {
            lr = 0.001,
            beta1 = 0.9,
            beta2 = 0.999,
            eps = 1e-8,
            weight_decay = 0.01,  
            max_grad_norm = 1.0
        } = options;
        
        this.params = params;
        this.lr = lr;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
        this.weight_decay = weight_decay;
        this.max_grad_norm = max_grad_norm;
        
        this.m = params.map(p => zeros(p.param.length, p.param[0].length || 1));
        this.v = params.map(p => zeros(p.param.length, p.param[0].length || 1));
        this.t = 0;
    }

    step() {
        this.t++;
        this.params.forEach((p, idx) => {
            // Gradient clipping (same as Adam)
            let grad_norm_sq = 0;
            for (let i = 0; i < p.param.length; i++) {
                for (let j = 0; j < (p.param[0].length || 1); j++) {
                    const grad_val = p.grad[i] && p.grad[i][j] !== undefined ? p.grad[i][j] : 0;
                    grad_norm_sq += grad_val * grad_val;
                }
            }
            const grad_norm = Math.sqrt(grad_norm_sq);
            const clip_scale = grad_norm > this.max_grad_norm ? this.max_grad_norm / grad_norm : 1.0;

            // AdamW update: weight decay applied separately
            for (let i = 0; i < p.param.length; i++) {
                for (let j = 0; j < (p.param[0].length || 1); j++) {
                    if (p.grad[i] && p.grad[i][j] !== undefined) {
                        const g = p.grad[i][j] * clip_scale;
                        
                        // Adam moments
                        this.m[idx][i][j] = this.beta1 * this.m[idx][i][j] + (1 - this.beta1) * g;
                        this.v[idx][i][j] = this.beta2 * this.v[idx][i][j] + (1 - this.beta2) * g * g;
                        
                        const mHat = this.m[idx][i][j] / (1 - Math.pow(this.beta1, this.t));
                        const vHat = this.v[idx][i][j] / (1 - Math.pow(this.beta2, this.t));
                        
                        // AdamW key difference: weight decay applied to weights, not gradients
                        p.param[i][j] -= this.lr * (
                            mHat / (Math.sqrt(vHat) + this.eps) + 
                            this.weight_decay * p.param[i][j]  // Decoupled weight decay
                        );
                    }
                }
            }
        });
    }
}

export class SGD{
    constructor(params, lr = 0.01, max_grad_norm = 1.0) {
        this.params = params;
        this.lr = lr;
        this.max_grad_norm = max_grad_norm; // Gradient Clipping
    }

    step() {
        this.params.forEach(p => {
            // Calculate gradient norm
            let grad_norm_sq = 0;
            let total_params = 0;

            for (let i = 0; i < p.param.length; i++){
                const row = p.param[i];
                for (let j = 0; j < (row.length || 1); j++) {
                    const grad_val = p.grad[i] && p.grad[i][j] !== undefined ? p.grad[i][j] : 0;
                    grad_norm_sq += grad_val * grad_val;
                    total_params++;
                }
            }

            const grad_norm = Math.sqrt(grad_norm_sq);

            // Apply gradient clipping if needed
            const clip_scale = grad_norm > this.max_grad_norm ? this.max_grad_norm / grad_norm : 1.0;

            // Update parameters with clipped gradients
            for (let i = 0; i < p.param.length; i++){
                const row = p.param[i];
                for (let j = 0; j < (row.length || 1); j++) {
                    if (p.grad[i] && p.grad[i][j] !== undefined){
                        p.param[i][j] -= this.lr * (p.grad[i][j] * clip_scale);
                    }
                }
            }
        });
    }
}


export class LION {
    constructor(params, options = {}) {
        this.params = params;
        
        const {
            lr = 0.0001,      // Lions typically uses smaller LR
            beta1 = 0.9,      // First moment decay
            beta2 = 0.99,     // Second moment decay  
            weight_decay = 0, // L2 regularization
            eps = 1e-8        // Numerical stability
        } = options;
        
        this.lr = lr;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.weight_decay = weight_decay;
        this.eps = eps;
        
        // Initialize momentums
        this.m = params.map(p => zeros(p.param.length, p.param[0].length || 1));
        this.t = 0;
    }

    step() {
        this.t++;
        
        this.params.forEach((p, idx) => {
            for (let i = 0; i < p.param.length; i++) {
                for (let j = 0; j < (p.param[0].length || 1); j++) {
                    if (p.grad[i] && p.grad[i][j] !== undefined) {
                        const grad = p.grad[i][j];
                        
                        // Update momentum: m_t = β1 * m_{t-1} + (1 - β1) * g_t
                        this.m[idx][i][j] = this.beta1 * this.m[idx][i][j] + (1 - this.beta1) * grad;
                        
                        // LIONS update: param = param - η * sign(m_t + β2 * g_t)
                        const update_term = this.m[idx][i][j] + this.beta2 * grad;
                        
                        // Get sign with epsilon for stability
                        let sign_val;
                        if (update_term > this.eps) sign_val = 1;
                        else if (update_term < -this.eps) sign_val = -1;
                        else sign_val = 0;
                        
                        let update = sign_val * this.lr;
                        
                        // Add weight decay if specified
                        if (this.weight_decay > 0) {
                            update += this.weight_decay * this.lr * p.param[i][j];
                        }
                        
                        p.param[i][j] -= update;
                    }
                }
            }
        });
    }

    zeroGrad() {
        this.params.forEach(p => {
            if (p.grad) {
                for (let i = 0; i < p.grad.length; i++) {
                    for (let j = 0; j < p.grad[i].length; j++) {
                        p.grad[i][j] = 0;
                    }
                }
            }
        });
    }
}

// ---------------------- Learning Rate Schedulers ----------------------
export class StepLR {
    constructor(optimizer, step_size, gamma=1.0) {
        this.optimizer = optimizer;
        this.step_size = step_size;
        this.gamma = gamma;
        this.last_epoch = 0;
        this.base_lr = optimizer.lr;
    }

    step() {
        this.last_epoch += 1;
        if (this.last_epoch % this.step_size === 0) {
            this.optimizer.lr *= this.gamma;
        }
    }

    get_lr() {
        return this.optimizer.lr;
		/* Do nothing else */
    }
}

export class LambdaLR {
    constructor(optimizer, lr_lambda) {
        this.optimizer = optimizer;
        this.lr_lambda = lr_lambda;
        this.last_epoch = 0;
        this.base_lr = optimizer.lr;
    }

    step() {
        this.last_epoch += 1;
        this.optimizer.lr = this.base_lr * this.lr_lambda(this.last_epoch);
    }

    get_lr() {
        return this.optimizer.lr;
		/* Do nothing else */
    }
}

// ---------------------- ReduceLROnPlateau Scheduler ----------------------
export class ReduceLROnPlateau {
    constructor(optimizer, options = {}) {
        this.optimizer = optimizer;
        
        // Destructure with defaults
        const {
            patience = 10,
            factor = 0.5, 
            min_lr = 1e-6,
            threshold = 1e-4,
            cooldown = 0,
            verbose = false
        } = options;
        
        this.patience = patience;
        this.factor = factor;
        this.min_lr = min_lr;
        this.threshold = threshold;
        this.cooldown = cooldown;
        this.verbose = verbose;
        
        // State tracking
        this.bestLoss = Infinity;
        this.wait = 0;
        this.cooldown_counter = 0;
        this.num_reductions = 0;
    }

    step(loss) {
        // Handle cooldown
        if (this.cooldown_counter > 0) {
            this.cooldown_counter--;
            return;
        }

        // Check if this is significant improvement (relative threshold)
        const improvement_needed = this.bestLoss * (1 - this.threshold);
        const is_better = loss < improvement_needed;
        
        if (is_better) {
            // Significant improvement - reset
            this.bestLoss = loss;
            this.wait = 0;
        } else {
            // No significant improvement
            this.wait += 1;
        }
        
        // Check if we've waited long enough
        if (this.wait >= this.patience) {
            this._reduce_lr();
            this.cooldown_counter = this.cooldown;
            this.wait = 0;
        }
    }

    _reduce_lr() {
        const old_lr = this.optimizer.lr;
        const new_lr = Math.max(old_lr * this.factor, this.min_lr);
        
        if (new_lr < old_lr) {
            this.optimizer.lr = new_lr;
            this.num_reductions++;
            
            if (this.verbose) {
                console.log(`ReduceLROnPlateau: reducing LR from ${old_lr} to ${new_lr}`);
            }
        }
    }

    get_last_lr() {
        return this.optimizer.lr;
    }

    reset() {
        this.bestLoss = Infinity;
        this.wait = 0;
        this.cooldown_counter = 0;
        this.num_reductions = 0;
    }
}

// ---------------------- ELU Activation ----------------------
export class ELU {
    constructor(alpha=1.0) {
        this.alpha = alpha;
        this.out = null;
    }

    forward(x) {
        this.out = x.map(row => 
            row.map(v => v > 0 ? v : this.alpha * (Math.exp(v) - 1))
        );
        return this.out;
    }

    backward(grad) {
        return grad.map((row, i) => 
            row.map((v, j) => 
                v * (this.out[i][j] > 0 ? 1 : this.alpha * Math.exp(this.out[i][j]))
            )
        );
    }
}

// ---------------------- Mish Activation ----------------------
export class Mish {
    constructor() {
        this.x = null;
    }

    forward(x) {
        this.x = x;
        return x.map(row => 
            row.map(v => {
                // Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))
                const softplus = Math.log(1 + Math.exp(v));
                return v * Math.tanh(softplus);
            })
        );
    }

    backward(grad) {
        return grad.map((row, i) => 
            row.map((v, j) => {
                const x_val = this.x[i][j];
                
                // Gradient of Mish: 
                // δ = ω * (4(x+1) + 4e^2x + e^3x + e^x(4x+6)) / (2e^x + e^2x + 2)^2
                // where ω = sech^2(softplus(x))
                
                const exp_x = Math.exp(x_val);
                const exp_2x = Math.exp(2 * x_val);
                const exp_3x = Math.exp(3 * x_val);
                const softplus = Math.log(1 + exp_x);
                
                const sech_softplus = 1 / Math.cosh(softplus);
                const numerator = 4 * (x_val + 1) + 4 * exp_2x + exp_3x + exp_x * (4 * x_val + 6);
                const denominator = Math.pow(2 * exp_x + exp_2x + 2, 2);
                
                const mish_grad = (sech_softplus * sech_softplus) * (numerator / denominator);
                return v * mish_grad;
            })
        );
    }
}

// ---------------------- SiLU Activation ----------------------
export class SiLU {
    constructor() {
        this.x = null;
    }

    forward(x) {
        this.x = x;
        return x.map(row => 
            row.map(v => v / (1 + Math.exp(-v)))  // x * sigmoid(x)
        );
    }

    backward(grad) {
        return grad.map((row, i) => 
            row.map((v, j) => {
                const x_val = this.x[i][j];
                const sigmoid = 1 / (1 + Math.exp(-x_val));
                return v * (sigmoid * (1 + x_val * (1 - sigmoid)));
            })
        );
    }
}


// ---------------------- BatchNorm2D ----------------------
export class BatchNorm2d {
    constructor(numFeatures, eps=1e-5, momentum=0.1, affine=true) {
        this.numFeatures = numFeatures;
        this.eps = eps;
        this.momentum = momentum;
        this.affine = affine;
        
        // Parameters
        if (affine) {
            this.weight = Array(numFeatures).fill(1);
            this.bias = Array(numFeatures).fill(0);
            this.gradWeight = Array(numFeatures).fill(0);
            this.gradBias = Array(numFeatures).fill(0);
        }
        
        // Running statistics
        this.runningMean = Array(numFeatures).fill(0);
        this.runningVar = Array(numFeatures).fill(1);
        
        // Training state
        this.training = true;
        this.x = null;
        this.xCentered = null;
        this.std = null;
    }

    forward(x) {
        // x shape: [batch, channels, height, width]
        this.x = x;
        const batchSize = x.length;
        const channels = x[0].length;
        
        if (this.training) {
            // Calculate mean per channel
            const means = Array(channels).fill(0);
            for (let b = 0; b < batchSize; b++) {
                for (let c = 0; c < channels; c++) {
                    const channelData = x[b][c];
                    let sum = 0;
                    for (let i = 0; i < channelData.length; i++) {
                        for (let j = 0; j < channelData[0].length; j++) {
                            sum += channelData[i][j];
                        }
                    }
                    means[c] += sum / (channelData.length * channelData[0].length);
                }
            }
            means.forEach((_, c) => means[c] /= batchSize);
            
            // Calculate variance per channel
            const variances = Array(channels).fill(0);
            for (let b = 0; b < batchSize; b++) {
                for (let c = 0; c < channels; c++) {
                    const channelData = x[b][c];
                    let sum = 0;
                    for (let i = 0; i < channelData.length; i++) {
                        for (let j = 0; j < channelData[0].length; j++) {
                            sum += Math.pow(channelData[i][j] - means[c], 2);
                        }
                    }
                    variances[c] += sum / (channelData.length * channelData[0].length);
                }
            }
            variances.forEach((_, c) => variances[c] /= batchSize);
            
            // Update running statistics
            for (let c = 0; c < channels; c++) {
                this.runningMean[c] = this.momentum * means[c] + (1 - this.momentum) * this.runningMean[c];
                this.runningVar[c] = this.momentum * variances[c] + (1 - this.momentum) * this.runningVar[c];
            }
            
            // Normalize
            this.xCentered = [];
            this.std = Array(channels).fill(0).map(() => []);
            
            const output = [];
            for (let b = 0; b < batchSize; b++) {
                const batchOut = [];
                for (let c = 0; c < channels; c++) {
                    const channelData = x[b][c];
                    const channelOut = zeros(channelData.length, channelData[0].length);
                    const channelCentered = zeros(channelData.length, channelData[0].length);
                    const channelStd = Math.sqrt(variances[c] + this.eps);
                    this.std[c].push(channelStd);
                    
                    for (let i = 0; i < channelData.length; i++) {
                        for (let j = 0; j < channelData[0].length; j++) {
                            channelCentered[i][j] = channelData[i][j] - means[c];
                            channelOut[i][j] = channelCentered[i][j] / channelStd;
                            
                            // Apply affine transformation if enabled
                            if (this.affine) {
                                channelOut[i][j] = channelOut[i][j] * this.weight[c] + this.bias[c];
                            }
                        }
                    }
                    
                    batchOut.push(channelOut);
                    if (b === 0) this.xCentered.push(channelCentered);
                    else this.xCentered[c] = addMatrices(this.xCentered[c], channelCentered);
                }
                output.push(batchOut);
            }
            
            return output;
        } else {
            // Inference mode - use running statistics
            const output = [];
            for (let b = 0; b < batchSize; b++) {
                const batchOut = [];
                for (let c = 0; c < channels; c++) {
                    const channelData = x[b][c];
                    const channelOut = zeros(channelData.length, channelData[0].length);
                    const channelStd = Math.sqrt(this.runningVar[c] + this.eps);
                    
                    for (let i = 0; i < channelData.length; i++) {
                        for (let j = 0; j < channelData[0].length; j++) {
                            channelOut[i][j] = (channelData[i][j] - this.runningMean[c]) / channelStd;
                            
                            // Apply affine transformation if enabled
                            if (this.affine) {
                                channelOut[i][j] = channelOut[i][j] * this.weight[c] + this.bias[c];
                            }
                        }
                    }
                    
                    batchOut.push(channelOut);
                }
                output.push(batchOut);
            }
            
            return output;
        }
    }

    backward(gradOutput) {
        if (!this.training) {
            throw new Error("Backward should only be called in training mode");
        }
        
        const batchSize = gradOutput.length;
        const channels = gradOutput[0].length;
        
        // Initialize gradients
        const gradInput = this.x.map(batch => 
            batch.map(channel => 
                zeros(channel.length, channel[0].length)
            )
        );
        
        if (this.affine) {
            this.gradWeight.fill(0);
            this.gradBias.fill(0);
        }
        
        for (let c = 0; c < channels; c++) {
            let sumGradWeight = 0;
            let sumGradBias = 0;
            
            for (let b = 0; b < batchSize; b++) {
                const channelGrad = gradOutput[b][c];
                const channelData = this.x[b][c];
                
                // Calculate gradients for bias and weight
                if (this.affine) {
                    for (let i = 0; i < channelGrad.length; i++) {
                        for (let j = 0; j < channelGrad[0].length; j++) {
                            sumGradBias += channelGrad[i][j];
                            sumGradWeight += channelGrad[i][j] * (this.xCentered[c][i][j] / this.std[c][b]);
                        }
                    }
                }
                
                // Calculate gradient for input
                const n = channelData.length * channelData[0].length;
                const stdInv = 1 / this.std[c][b];
                
                for (let i = 0; i < channelGrad.length; i++) {
                    for (let j = 0; j < channelGrad[0].length; j++) {
                        let grad = channelGrad[i][j];
                        
                        if (this.affine) {
                            grad *= this.weight[c];
                        }
                        
                        grad *= stdInv;
                        gradInput[b][c][i][j] = grad;
                    }
                }
            }
            
            if (this.affine) {
                this.gradWeight[c] = sumGradWeight / batchSize;
                this.gradBias[c] = sumGradBias / batchSize;
            }
        }
        
        return gradInput;
    }

    parameters() {
        if (!this.affine) return [];
        return [
            { param: [this.weight], grad: [this.gradWeight] },
            { param: [this.bias], grad: [this.gradBias] }
        ];
    }

    train() { this.training = true; }
    eval() { this.training = false; }
}

// ---------------------- Model Save/Load ----------------------
export function saveModel(model){
    if(!(model instanceof Sequential)) throw new Error("saveModel supports only Sequential");
    const weights=model.layers.map(layer=>({weights:layer.W||null,biases:layer.b||null}));
    return JSON.stringify(weights);
	/* Didn't expect this to work /: */
}

export function loadModel(model,json){
    if(!(model instanceof Sequential)) throw new Error("loadModel supports only Sequential");
    const weights=JSON.parse(json);
    model.layers.forEach((layer,i)=>{
        if(layer.W && weights[i].weights) layer.W=weights[i].weights;
        if(layer.b && weights[i].biases) layer.b=weights[i].biases;
    });
	/* Didn't expect this to work /: */
}

// ---------------------- Advanced Utils ----------------------
export function flattenBatch(batch){ return batch.flat(2); }
export function stack(tensors){ return tensors.map(t=>t.data); }
export function eye(n){ return Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>i===j?1:0)); }
export function concat(a,b,axis=0){ /* concat along axis */ if(axis===0) return [...a,...b]; if(axis===1) return a.map((row,i)=>[...row,...b[i]]); }
export function reshape(tensor, rows, cols) {
    let flat = tensor.data.flat(); // flatten first
    if(flat.length < rows*cols) throw new Error("reshape size mismatch");
    const out = Array.from({length: rows}, (_, i) =>
        flat.slice(i*cols, i*cols + cols)
    );
    return out;
}