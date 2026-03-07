
import { 
    fu_tensor, fu_add, fu_mul, fu_matmul, fu_sum, fu_mean, 
    fu_relu, fu_sigmoid, fu_tanh, fu_softmax, fu_flatten, fu_reshape 
} from '../src/jstorch.js';

function testAllFuFunctions() {
    console.log("TESTING ALL FU_FUNCTIONS\n");
    
    // Test 1: fu_tensor
    console.log("1. fu_tensor");
    const t1 = fu_tensor([[1, 2], [3, 4]]);
    console.log("", t1.data);
    
    // Test 2: fu_add
    console.log("\n2. fu_add");
    const a = fu_tensor([[1, 2]]);
    const b = fu_tensor([[3, 4]]);
    const c = fu_add(a, b);
    console.log("", a.data, "+", b.data, "=", c.data);
    
    // Test 3: fu_mul
    console.log("\n3. fu_mul");
    const d = fu_mul(a, b);
    console.log("", a.data, "*", b.data, "=", d.data);
    
    // Test 4: fu_matmul
    console.log("\n4. fu_matmul");
    const e = fu_tensor([[1, 2]]);
    const f = fu_tensor([[3], [4]]);
    const g = fu_matmul(e, f);
    console.log("matmul =", g.data);
    
    // Test 5: fu_sum & fu_mean
    console.log("\n5. fu_sum & fu_mean");
    const h = fu_tensor([[1, 2], [3, 4]]);
    const sum = fu_sum(h);
    const mean = fu_mean(h);
    console.log("sum =", sum.data, "mean =", mean.data);
    
    // Test 6: fu_relu
    console.log("\n6. fu_relu");
    const i = fu_tensor([[-1, 0], [1, 2]]);
    const relu = fu_relu(i);
    console.log("relu =", relu.data);
    
    // Test 7: fu_sigmoid
    console.log("\n7. fu_sigmoid");
    const sigmoid = fu_sigmoid(i);
    console.log("sigmoid =", sigmoid.data);
    
    // Test 8: fu_tanh
    console.log("\n8. fu_tanh");
    const tanh = fu_tanh(i);
    console.log("tanh =", tanh.data);
    
    // Test 9: fu_softmax
    console.log("\n9. fu_softmax");
    const j = fu_tensor([[1, 2, 3]]);
    const softmax = fu_softmax(j);
    console.log("softmax =", softmax.data);
    
    // Test 10: fu_flatten & fu_reshape
    console.log("\n10. fu_flatten & fu_reshape");
    const k = fu_tensor([[1, 2], [3, 4]]);
    const flat = fu_flatten(k);
    const reshaped = fu_reshape(flat, 1, 4);
    console.log("flatten =", flat.data);
    console.log("reshape =", reshaped.data);
}

testAllFuFunctions();