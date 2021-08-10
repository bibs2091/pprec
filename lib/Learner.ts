import * as tf from '@tensorflow/tfjs-node'
import { DataBlock } from './DataBlock'
import { SigmoidRange } from './SigmoidRange'
export class Learner {
    itemsNum: number;
    usersNum: number;
    learningRate: number;
    lossFunc: string;
    optimizer: tf.Optimizer;
    trainingDataset: tf.data.Dataset<any>;
    validationDataset: tf.data.Dataset<any>;
    userInputLayer; userEmbeddingLayer; userEmbeddingLayerOutput; itemInputLayer; itemEmbeddingLayer; itemEmbeddingLayerOutput; dotLayer;
    sigmoidLayer;
    model: tf.LayersModel;
    constructor(dataBlock: DataBlock, usersNum: number, itemsNum: number, learningRate: number = 1e-2, lossFunc: string = "meanSquaredError", optimizerName: string = "adam", embeddingOutputSize: number = 5, weightDecay: number = 0, options?: object) {
        this.itemsNum = itemsNum;
        this.usersNum = usersNum;
        this.lossFunc = lossFunc;
        this.learningRate = learningRate;
        this.trainingDataset = dataBlock.trainingDataset;
        this.validationDataset = dataBlock.validationDataset;

        this.createModel(embeddingOutputSize, weightDecay, dataBlock.ratingRange);
        this.setOptimizer(optimizerName);
    }

    createModel(embeddingOutputSize: number, weightDecay: number, ratingRange: null | number[]) {
        this.userInputLayer = tf.input({ shape: [1], dtype: "int32", name: "user" });
        this.userEmbeddingLayer = tf.layers.embedding({
            inputDim: this.usersNum + 1,
            outputDim: embeddingOutputSize,
            inputLength: 1,
            name: "userEmbeddingLayer",
            embeddingsRegularizer: tf.regularizers.l2({ l2: weightDecay })
        }).apply(this.userInputLayer)
        this.userEmbeddingLayerOutput = tf.layers.flatten({ name: "flat1" }).apply(this.userEmbeddingLayer);

        this.itemInputLayer = tf.input({ shape: [1], dtype: "int32", name: "item" });
        this.itemEmbeddingLayer = tf.layers.embedding({
            inputDim: this.itemsNum + 1,
            outputDim: embeddingOutputSize,
            inputLength: 1,
            name: "itemEmbeddingLayer",
            embeddingsRegularizer: tf.regularizers.l2({ l2: weightDecay })
        }).apply(this.itemInputLayer);
        this.itemEmbeddingLayerOutput = tf.layers.flatten({ name: "flat2" }).apply(this.itemEmbeddingLayer);

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

    setOptimizer(optimizerName: string) {
        switch (optimizerName) {
            case "adam":
                this.optimizer = tf.train.adam(this.learningRate);
                break;
            case "sgd":
                this.optimizer = tf.train.sgd(this.learningRate);
                break;
            case "rmsprop":
                this.optimizer = tf.train.rmsprop(this.learningRate);
                break;
        }
        this.model.compile({
            optimizer: this.optimizer,
            loss: this.lossFunc
        });
    }

    fit(epochs: number = 1) {
        return this.model.fitDataset(this.trainingDataset, {
            validationData: this.validationDataset,
            epochs: epochs,
        })
    }


    recommendItem(userId: number) {
        let toPredict = [tf.fill([this.itemsNum, 1], userId), tf.range(0, this.itemsNum).reshape([-1, 1])]
        return (this.model.predictOnBatch(toPredict) as tf.Tensor).argMax();
    }

    save(path: string) {
        return this.model.save('file://' + path);
    }

    load(path: string) {
        return tf.loadLayersModel('file://' + path);
    }
}