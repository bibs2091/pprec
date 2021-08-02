import * as tf from '@tensorflow/tfjs-node'
import { DataBlock } from './DataBlock'

class learner {
    itemsNum: number;
    usersNum: number;
    learningRate: number;
    lossFunc: string;
    optimizer: tf.Optimizer;
    dataSet: tf.data.Dataset<any>;
    validationDataset: tf.data.Dataset<any>;
    userInputLayer; userEmbeddingLayer; userEmbeddingLayerOutput; itemInputLayer; itemEmbeddingLayer; itemEmbeddingLayerOutput; dotLayer;
    model: tf.LayersModel;
    constructor(dataBlock: DataBlock, itemsNum: number, usersNum: number, learningRate: number, lossFunc: string, optimizerName: string, embeddingOutputSize: number = 5, options: object | null) {
        this.itemsNum = itemsNum;
        this.usersNum = usersNum;
        this.lossFunc = lossFunc;
        this.learningRate = learningRate;
        this.dataSet = dataBlock.dataSet;

        this.createModel(embeddingOutputSize);
        this.setOptimizer(optimizerName)


    }

    createModel(embeddingOutputSize: number) {
        this.userInputLayer = tf.input({ shape: [1], dtype: "int32", name: "user" });
        this.userEmbeddingLayer = tf.layers.embedding({
            inputDim: this.usersNum + 1,
            outputDim: embeddingOutputSize,
            inputLength: 1,
            name: "userEmbeddingLayer",
        }).apply(this.userInputLayer)
        this.userEmbeddingLayerOutput = tf.layers.flatten({ name: "flat1" }).apply(this.userEmbeddingLayer);

        this.itemInputLayer = tf.input({ shape: [1], dtype: "int32", name: "movie" });
        this.itemEmbeddingLayer = tf.layers.embedding({
            inputDim: this.itemsNum + 1,
            outputDim: embeddingOutputSize,
            inputLength: 1,
            name: "itemEmbeddingLayer",
        }).apply(this.itemInputLayer);
        this.itemEmbeddingLayerOutput = tf.layers.flatten({ name: "flat2" }).apply(this.itemEmbeddingLayer);
        this.dotLayer = tf.layers.dot({ axes: -1, name: "rating" }).apply([this.userEmbeddingLayerOutput, this.itemEmbeddingLayerOutput]);
        this.model = tf.model({ inputs: [this.userInputLayer, this.itemInputLayer], outputs: this.dotLayer });
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
}