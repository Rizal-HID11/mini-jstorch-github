// package root

// provide JST in browser global scope
import * as JST from './src/jstorch.js';

if (typeof window !== 'undefined') {
    window.JST = JST; // Global JST (JSTorch) object
}

export * from './src/jstorch.js';