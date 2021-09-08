import { MatrixFactorization } from './MatrixFactorization';
import * as tf from '@tensorflow/tfjs-node'
import { DataBlock } from './DataBlock'
import { cosineSimilarity, euclideandistance } from './utils'
import { ValueError } from './errors'
import { io } from '@tensorflow/tfjs-core';

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
    model: tf.LayersModel;
    MFC: MatrixFactorization;
    ratingRange?: number[];
    optimizerName: string;
    dataBlock: DataBlock;

    constructor(dataBlock: DataBlock, learningRate: number = 1e-2, lossFunc: string = "meanSquaredError", optimizerName: string = "adam", embeddingOutputSize: number = 5, weightDecay: number = 0, options?: object) {
        this.dataBlock = dataBlock;
        this.itemsNum = this.dataBlock.datasetInfo.itemsNum;
        this.usersNum = this.dataBlock.datasetInfo.usersNum;
        this.lossFunc = lossFunc;
        this.learningRate = learningRate;
        this.embeddingOutputSize = embeddingOutputSize;
        this.ratingRange = this.dataBlock.ratingRange
        this.MFC = new MatrixFactorization(this.usersNum, this.itemsNum, this.embeddingOutputSize, weightDecay, this.ratingRange);
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
            default:
                throw new ValueError(`${optimizerName} optimzer does not exist in pprec. Only adam, sgd, and rmsprop are supported (lower case)`);

        }
        this.model.compile({
            optimizer: this.optimizer,
            loss: this.lossFunc
        });
    }

    /*
    To train the model in a number of epoches
    */
    fit(epochs: number = 1): Promise<tf.History>   {
        return this.model.fitDataset(this.dataBlock.trainingDataset, {
            validationData: this.dataBlock.validationDataset,
            epochs: epochs,
        })
    }

    /*
        To recommend an Item for a user given their ID
    */
    recommendItem(userId: number): tf.Tensor  {
        let toPredict = [tf.fill([this.itemsNum, 1], userId), tf.range(0, this.itemsNum).reshape([-1, 1])]
        return (this.model.predictOnBatch(toPredict) as tf.Tensor).argMax();
    }

    addRating(userId: number, itemId: number, rating: number, train: boolean = true): void | Promise<tf.History> {
        let toAdd = tf.data.array([{ xs: { user: tf.tensor2d([[userId]]), item: tf.tensor2d([[itemId]]) }, ys: { rating: tf.tensor1d([rating]) } },])
        this.dataBlock.trainingDataset = this.dataBlock.trainingDataset.concatenate(toAdd);

        if (train) {
            return this.model.fitDataset(toAdd, {
                epochs: 1,
                verbose: 0
            })
        }
    }


    async addRatingSync(userId: number, itemId: number, rating: number, train: boolean = true) {
        let toAdd = tf.data.array([{ xs: { user: tf.tensor2d([[userId]]), item: tf.tensor2d([[itemId]]) }, ys: { rating: tf.tensor1d([rating]) } },])
        this.dataBlock.trainingDataset = this.dataBlock.trainingDataset.concatenate(toAdd);

        if (train) {
            await this.model.fitDataset(toAdd, {
                epochs: 1,
                verbose: 0
            })
        }
    }


    /*
        To add a new user embedding in the model.
        The embedding is generated based on the mean of the other users latent factors.
    */
    newUser(): number {
        this.usersNum += 1
        let userEmbeddingWeight = this.MFC.userEmbeddingLayer.getWeights()[0];
        userEmbeddingWeight = tf.concat([userEmbeddingWeight, userEmbeddingWeight.mean(0).reshape([1, this.embeddingOutputSize])]);
        this.MFC = new MatrixFactorization(this.usersNum, this.itemsNum, this.embeddingOutputSize, 0, this.ratingRange, [userEmbeddingWeight]);
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
        itemEmbeddingWeight = tf.concat([itemEmbeddingWeight, itemEmbeddingWeight.mean(0).reshape([1, this.embeddingOutputSize])]);
        this.MFC = new MatrixFactorization(this.usersNum, this.itemsNum, this.embeddingOutputSize, 0, this.ratingRange, itemEmbeddingWeight = [itemEmbeddingWeight]);
        this.model = this.MFC.model;
        this.setOptimizer(this.optimizerName);
        return this.itemsNum //the new user ID
    }



    /*
       To retrieve the k similar users of a user 
    */
    mostSimilarUsers(id: number, k = 10): number[] {
        if (k < 1) throw new ValueError(`the k in mostSimilarUsers >= 1`);
        let userEmbeddingWeight = this.MFC.userEmbeddingLayer.getWeights()[0];
        let similarity = cosineSimilarity(userEmbeddingWeight, userEmbeddingWeight.slice(id, 1))
        let { values, indices } = tf.topk(similarity, k + 1);
        let indicesArray = (indices.arraySync() as number[])
        return indicesArray.filter((e: number) => e !== id)
    }

    /*
       To retrieve the k similar items of an item 
    */
    mostSimilarItems(id: number, k = 10): number[] {
        if (k < 1) throw new ValueError(`the k in mostSimilarItems >= 1`);
        let itemEmbeddingWeight = this.MFC.itemEmbeddingLayer.getWeights()[0];
        let similarity = euclideandistance(itemEmbeddingWeight, itemEmbeddingWeight.slice(id, 1))
        let { values, indices } = tf.topk(similarity, k + 1);
        let indicesArray = (indices.arraySync() as number[])
        return indicesArray.filter((e: number) => e !== id)
    }

    /*
       To save the architecture and the weights of the model in a given path
    */
    save(path: string): Promise<io.SaveResult> {
        return this.model.save('file://' + path);
    }

    load(path: string): Promise<tf.LayersModel> {
        return tf.loadLayersModel('file://' + path);
    }
}