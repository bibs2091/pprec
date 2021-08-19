import { Embedding } from '@tensorflow/tfjs-layers/dist/layers/embeddings';
import * as tf from '@tensorflow/tfjs-node'
import { SigmoidRange } from './SigmoidRange'

export class MatrixFactorization {
    userInputLayer; userEmbeddingLayer: Embedding; userEmbeddingLayerOutput; itemInputLayer; itemEmbeddingLayer; itemEmbeddingLayerOutput; dotLayer;
    model: tf.LayersModel;
    sigmoidLayer;

    constructor(usersNum: number, itemsNum: number, embeddingOutputSize: number, weightDecay: number, ratingRange?: number[], userEmbeddingWeights?, itemEmbeddingWeights?) {
        this.userInputLayer = tf.input({ shape: [1], dtype: "int32", name: "user" });
        this.userEmbeddingLayer = tf.layers.embedding({
            inputDim: usersNum + 1,
            outputDim: embeddingOutputSize,
            inputLength: 1,
            name: "userEmbeddingLayer",
            embeddingsRegularizer: tf.regularizers.l2({ l2: weightDecay })
        })

        if (userEmbeddingWeights != null) {
            console.log(userEmbeddingWeights);
            this.userEmbeddingLayer.setWeights(userEmbeddingWeights)
        }


        this.userEmbeddingLayerOutput = tf.layers.flatten({ name: "flat1" }).apply(this.userEmbeddingLayer.apply(this.userInputLayer));

        this.itemInputLayer = tf.input({ shape: [1], dtype: "int32", name: "item" });
        this.itemEmbeddingLayer = tf.layers.embedding({
            inputDim: itemsNum + 1,
            outputDim: embeddingOutputSize,
            inputLength: 1,
            name: "itemEmbeddingLayer",
            embeddingsRegularizer: tf.regularizers.l2({ l2: weightDecay })
        })
        this.itemEmbeddingLayerOutput = tf.layers.flatten({ name: "flat2" }).apply(this.itemEmbeddingLayer.apply(this.itemInputLayer));

        // if user did not specify a range for the ratings
        if (ratingRange == null) {
            this.dotLayer = tf.layers.dot({ axes: -1, name: "rating" }).apply([this.userEmbeddingLayerOutput, this.itemEmbeddingLayerOutput]);
            this.model = tf.model({ inputs: [this.userInputLayer, this.itemInputLayer], outputs: this.dotLayer });
        }

        // if user did specify a range for the ratings high and low
        else {
            this.dotLayer = tf.layers.dot({ axes: -1, name: "dot" }).apply([this.userEmbeddingLayerOutput, this.itemEmbeddingLayerOutput]);
            this.sigmoidLayer = new SigmoidRange({ high: ratingRange[1], low: ratingRange[0], name: "rating" }).apply(this.dotLayer)
            this.model = tf.model({ inputs: [this.userInputLayer, this.itemInputLayer], outputs: this.sigmoidLayer });
        }
    }
}
