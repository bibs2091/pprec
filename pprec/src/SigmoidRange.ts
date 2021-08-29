import * as tf from '@tensorflow/tfjs-node'

/*
   SigmoidRange is a custom layer to bound the output of the model in a certain range to speed up the training speed.
*/
export class SigmoidRange extends tf.layers.Layer {
    high: number;
    low: number;
    constructor(config) {
        super(config);
        this.high = config.high;
        this.low = config.low;
    }


    call(inputs: tf.Tensor | tf.Tensor[], kwargs: any): tf.Tensor | tf.Tensor[] {
        const x = tf.sigmoid(getExactlyOneTensor(inputs)).mul(this.high - this.low).add(this.low);
        return x
    }

 
    getConfig(): tf.serialization.ConfigDict {
        const config: tf.serialization.ConfigDict = { high: this.high, low: this.low };
        const baseConfig = super.getConfig();
        Object.assign(config, baseConfig);
        return config;
    }

  
    static get className() {
        return 'SigmoidRange';
    }
}
tf.serialization.registerClass(SigmoidRange);



function getExactlyOneTensor(xs: tf.Tensor|tf.Tensor[]): tf.Tensor {
 let x: tf.Tensor;
 if (Array.isArray(xs)) {
   if (xs.length !== 1) {
   }
   x = xs[0];
 } else {
   x = xs;
 }
 return x;
}