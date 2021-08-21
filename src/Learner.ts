import { MatrixFactorization } from './MatrixFactorization';
import * as tf from '@tensorflow/tfjs-node'
import { DataBlock } from './DataBlock'

/*
    Learner is an api which allows you to create, edit and train your model in few lines.
*/
export class Learner {
    itemsNum: number;
    usersNum: number;
    learningRate: number;
    lossFunc: string;
    embeddingOutputSize: number;
    optimizer: tf.Optimizer;
    trainingDataset: tf.data.Dataset<any>;
    validationDataset: tf.data.Dataset<any>;
    model: tf.LayersModel;
    MFC: MatrixFactorization;
    ratingRange?: number[];
    optimizerName: string;
    
    constructor(dataBlock: DataBlock, learningRate: number = 1e-2, lossFunc: string = "meanSquaredError", optimizerName: string = "adam", embeddingOutputSize: number = 5, weightDecay: number = 0, options?: object) {
        this.itemsNum = dataBlock.datasetInfo.itemsNum;
        this.usersNum = dataBlock.datasetInfo.usersNum;
        this.lossFunc = lossFunc;
        this.learningRate = learningRate;
        this.trainingDataset = dataBlock.trainingDataset;
        this.validationDataset = dataBlock.validationDataset;
        this.embeddingOutputSize = embeddingOutputSize;
        this.ratingRange = dataBlock.ratingRange
        this.MFC = new MatrixFactorization(this.usersNum, this.itemsNum, this.embeddingOutputSize, weightDecay, this.ratingRange );
        this.model = this.MFC.model;
        this.optimizerName = optimizerName
        this.setOptimizer(this.optimizerName);
    }

    
    /*
    To set the right optimizer for the model
    */
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

    /*
    To train the model in a number of epoches
    */
    fit(epochs: number = 1) {
        return this.model.fitDataset(this.trainingDataset, {
            validationData: this.validationDataset,
            epochs: epochs,
        })
    }

    /*
        To recommend an Item for a user given their ID
    */
    recommendItem(userId: number) {
        let toPredict = [tf.fill([this.itemsNum, 1], userId), tf.range(0, this.itemsNum).reshape([-1, 1])]
        return (this.model.predictOnBatch(toPredict) as tf.Tensor).argMax();
    }

    /*
        To add a new user embedding in the model.
        The embedding is generated based on the mean of the other users latent factors.
    */
    newUser() {
        this.usersNum += 1
        let userEmbeddingWeight = this.MFC.userEmbeddingLayer.getWeights()[0];
        userEmbeddingWeight = tf.concat([userEmbeddingWeight,userEmbeddingWeight.mean(0).reshape([1,this.embeddingOutputSize])]);
        this.MFC = new MatrixFactorization(this.usersNum, this.itemsNum, this.embeddingOutputSize, 0, this.ratingRange,[userEmbeddingWeight]);
        this.model = this.MFC.model;
        this.setOptimizer(this.optimizerName);
        return this.usersNum //the new user ID
    }

    /*
        To add a new item embedding in the model.
        The embedding is generated based on the mean of the other item latent factors.
    */
    newItem() {
        this.itemsNum += 1
        let itemEmbeddingWeight = this.MFC.itemEmbeddingLayer.getWeights()[0];
        itemEmbeddingWeight = tf.concat([itemEmbeddingWeight,itemEmbeddingWeight.mean(0).reshape([1,this.embeddingOutputSize])]);
        this.MFC = new MatrixFactorization(this.usersNum, this.itemsNum, this.embeddingOutputSize, 0, this.ratingRange,itemEmbeddingWeight=[itemEmbeddingWeight]);
        this.model = this.MFC.model;
        this.setOptimizer(this.optimizerName);
        return this.itemsNum //the new user ID
    }

    /*
       To save the architecture and the weights of the model in a given path
    */
    save(path: string) {
        return this.model.save('file://' + path);
    }

    load(path: string) {
        return tf.loadLayersModel('file://' + path);
    }
}