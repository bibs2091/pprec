import { Embedding } from '@tensorflow/tfjs-layers/dist/layers/embeddings';
import * as tf from '@tensorflow/tfjs-node'
import { SigmoidRange } from './SigmoidRange'


/**
   MatrixFactorization allows you to create a matrix factorization model to be used in Learner.
*/
export class MatrixFactorization {
    userInputLayer; userEmbeddingLayer: Embedding; userEmbeddingLayerOutput; itemInputLayer; itemEmbeddingLayer; itemEmbeddingLayerOutput; dotLayer;
    model: tf.LayersModel;
    sigmoidLayer;

    constructor(usersNum: number, itemsNum: number, embeddingOutputSize: number, weightDecay: number, ratingRange?: number[], userEmbeddingWeights?, itemEmbeddingWeights?) {
        this.userInputLayer = tf.input({ shape: [1], dtype: "int32", name: "user" });
        this.itemInputLayer = tf.input({ shape: [1], dtype: "int32", name: "item" });


        if (userEmbeddingWeights == null) {
            this.userEmbeddingLayer = tf.layers.embedding({
                inputDim: usersNum + 1,
                outputDim: embeddingOutputSize,
                inputLength: 1,
                name: "userEmbeddingLayer",
                embeddingsRegularizer: tf.regularizers.l2({ l2: weightDecay })
            })
        }
        else {
            this.userEmbeddingLayer = tf.layers.embedding({
                inputDim: usersNum + 1,
                outputDim: embeddingOutputSize,
                inputLength: 1,
                name: "userEmbeddingLayer",
                embeddingsRegularizer: tf.regularizers.l2({ l2: weightDecay }),
                weights: userEmbeddingWeights
            })
        }

        if (itemEmbeddingWeights == null) {
            this.itemEmbeddingLayer = tf.layers.embedding({
                inputDim: itemsNum,
                outputDim: embeddingOutputSize,
                inputLength: 1,
                name: "itemEmbeddingLayer",
                embeddingsRegularizer: tf.regularizers.l2({ l2: weightDecay })
            })
        }
        else {
            this.itemEmbeddingLayer = tf.layers.embedding({
                inputDim: itemsNum,
                outputDim: embeddingOutputSize,
                inputLength: 1,
                name: "itemEmbeddingLayer",
                embeddingsRegularizer: tf.regularizers.l2({ l2: weightDecay }),
                weights: itemEmbeddingWeights
            })
        }


        this.userEmbeddingLayerOutput = tf.layers.flatten({ name: "flat1" }).apply(this.userEmbeddingLayer.apply(this.userInputLayer));
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
        // this.model.summary()
    }
}
