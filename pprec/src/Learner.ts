import { MatrixFactorization } from './MatrixFactorization';
import * as tf from '@tensorflow/tfjs-node'
import { DataBlock } from './DataBlock'
import { cosineSimilarity, euclideandistance } from './utils'
import { ValueError, NonExistance } from './errors'
import { io } from '@tensorflow/tfjs-core';

interface optionsLearner {
    learningRate: number; embeddingOutputSize?: number; lossFunc?: string; optimizerName?: string; l2Labmda?: number
}

/**
    Learner is an api which allows you to create, edit and train your model in few lines.
*/
export class Learner {
    itemsNum?: number;
    usersNum?: number;
    learningRate: number;
    lossFunc: string;
    embeddingOutputSize?: number;
    optimizer: tf.Optimizer;
    model: tf.LayersModel;
    MFC?: MatrixFactorization;
    ratingRange?: number[];
    optimizerName: string;
    dataBlock?: DataBlock;
    l2Labmda?: number;
    constructor(dataBlock?: DataBlock, options?: optionsLearner) {
        if (dataBlock != null && options != null) {
            this.dataBlock = dataBlock;
            this.itemsNum = this.dataBlock.datasetInfo.itemsNum;
            this.usersNum = this.dataBlock.datasetInfo.usersNum;
            this.ratingRange = this.dataBlock.ratingRange
            this.learningRate = options.learningRate;

            if (options.lossFunc == null) this.lossFunc = "meanSquaredError";
            else this.lossFunc = options.lossFunc;

            if (options.embeddingOutputSize == null) this.embeddingOutputSize = 25;
            else this.embeddingOutputSize = options.embeddingOutputSize;

            if (options.l2Labmda == null) this.l2Labmda = 0;
            else this.l2Labmda = options.l2Labmda;

            this.MFC = new MatrixFactorization(this.usersNum, this.itemsNum, this.embeddingOutputSize, this.l2Labmda, this.ratingRange);
            this.model = this.MFC.model;

            if (options.optimizerName == null) this.optimizerName = "adam";
            else this.optimizerName = options.optimizerName;
            this.setOptimizer(this.optimizerName);
        }
        else {
            this.lossFunc = "meanSquaredError";
            this.optimizerName = "adam";
        }

    }


    /**
     * Setting an optimizer for leaner
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

    /**
    To train the model in a number of epoches
    */
    fit(epochs: number = 1): Promise<tf.History> {
        if (this.dataBlock == null)
            throw new NonExistance(`No datablock to train on, please provoid a proper DataBlock `);

        if (this.model == null)
            throw new NonExistance(`No model to train, please provoid a proper model`);

        return this.model.fitDataset(this.dataBlock.trainingDataset, {
            validationData: this.dataBlock.validationDataset,
            epochs: epochs,
        })
    }

    /**
        To recommend k items for a user given their ID
    */
    recommendItems(userId: number, k:number): number[] {
        if (this.itemsNum == null)
            throw new NonExistance(
                `itemsNum does not exist, this is maybe because you did not feed Learner a DataBlock`
            );

        if (this.model == null)
            throw new NonExistance(`No model to train, please provoid a proper model`);

        let toPredict = [tf.fill([this.itemsNum, 1], userId), tf.range(0, this.itemsNum).reshape([-1, 1])]
        const {values, indices} = tf.topk((this.model.predictOnBatch(toPredict) as tf.Tensor).flatten(), k);

        return (indices.arraySync() as number[]);
    }

    addRating(userId: number, itemId: number, rating: number, train: boolean = true): void | Promise<tf.History> {

        if (this.dataBlock == null)
            throw new NonExistance(`No datablock to train on, please provoid a proper DataBlock `);


        let toAdd = tf.data.array([{ xs: { user: tf.tensor2d([[userId]]), item: tf.tensor2d([[itemId]]) }, ys: { rating: tf.tensor1d([rating]) } },])
        this.dataBlock.trainingDataset = this.dataBlock.trainingDataset.concatenate(toAdd);
        this.dataBlock.datasetInfo.size++;
        if (train) {

            if (this.model == null)
                throw new NonExistance(`No model to train, please provoid a proper model`);

            return this.model.fitDataset(toAdd, {
                epochs: 1,
                verbose: 0
            })
        }
    }


    async addRatingSync(userId: number, itemId: number, rating: number, train: boolean = true) {

        if (this.dataBlock == null)
            throw new NonExistance(`No datablock to train on, please provoid a proper DataBlock `);

        let toAdd = tf.data.array([{ xs: { user: tf.tensor2d([[userId]]), item: tf.tensor2d([[itemId]]) }, ys: { rating: tf.tensor1d([rating]) } },])
        this.dataBlock.trainingDataset = this.dataBlock.trainingDataset.concatenate(toAdd);
        this.dataBlock.datasetInfo.size++;

        if (train) {
            await this.model.fitDataset(toAdd, {
                epochs: 1,
                verbose: 0
            })
        }
    }


    /**
        To add a new user embedding in the model.
        The embedding is generated based on the mean of the other users latent factors.
    */
    newUser(): number {
        if (this.usersNum == null)
            throw new NonExistance(
                `usersNum does not exist, this is maybe because you did not feed Learner a DataBlock`
            );

        if (this.itemsNum == null)
            throw new NonExistance(
                `itemsNum does not exist, this is maybe because you did not feed Learner a DataBlock`
            );

        if (this.embeddingOutputSize == null)
            throw new NonExistance(`embeddingOutputSize does not exist`);

        if (this.model == null || this.MFC == null)
            throw new NonExistance(`No model to train, please provoid a proper model`);

        this.usersNum += 1
        let userEmbeddingWeight = this.MFC.userEmbeddingLayer.getWeights()[0];
        userEmbeddingWeight = tf.concat([userEmbeddingWeight, userEmbeddingWeight.mean(0).reshape([1, this.embeddingOutputSize])]);
        this.MFC = new MatrixFactorization(this.usersNum, this.itemsNum, this.embeddingOutputSize, 0, this.ratingRange, [userEmbeddingWeight]);
        this.model = this.MFC.model;
        this.setOptimizer(this.optimizerName);
        return this.usersNum //the new user ID
    }

    /**
        To add a new item embedding in the model.
        The embedding is generated based on the mean of the other item latent factors.
    */
    newItem() {
        if (this.usersNum == null)
            throw new NonExistance(
                `usersNum does not exist, this is maybe because you did not feed Learner a DataBlock`
            );

        if (this.itemsNum == null)
            throw new NonExistance(
                `itemsNum does not exist, this is maybe because you did not feed Learner a DataBlock`
            );

        if (this.embeddingOutputSize == null)
            throw new NonExistance(`embeddingOutputSize does not exist`);

        if (this.model == null || this.MFC == null)
            throw new NonExistance(`No model to train, please provoid a proper model`);

        this.itemsNum += 1
        let itemEmbeddingWeight = this.MFC.itemEmbeddingLayer.getWeights()[0];
        itemEmbeddingWeight = tf.concat([itemEmbeddingWeight, itemEmbeddingWeight.mean(0).reshape([1, this.embeddingOutputSize])]);
        this.MFC = new MatrixFactorization(this.usersNum, this.itemsNum, this.embeddingOutputSize, 0, this.ratingRange, itemEmbeddingWeight = [itemEmbeddingWeight]);
        this.model = this.MFC.model;
        this.setOptimizer(this.optimizerName);
        return this.itemsNum //the new user ID
    }



    /**
       To retrieve the k similar users of a user 
    */
    mostSimilarUsers(id: number, k = 10): number[] {

        if (this.model == null || this.MFC == null)
            throw new NonExistance(`No model to train, please provoid a proper model`);

        if (k < 1) throw new ValueError(`the k in mostSimilarUsers >= 1`);
        let userEmbeddingWeight = this.MFC.userEmbeddingLayer.getWeights()[0];
        let similarity = cosineSimilarity(userEmbeddingWeight, userEmbeddingWeight.slice(id, 1))
        let { values, indices } = tf.topk(similarity, k + 1);
        let indicesArray = (indices.arraySync() as number[])
        return indicesArray.filter((e: number) => e !== id)
    }

    /**
       To retrieve the k similar items of an item 
    */
    mostSimilarItems(id: number, k = 10): number[] {
        if (this.model == null || this.MFC == null)
            throw new NonExistance(`No model to train, please provoid a proper model`);

        if (k < 1) throw new ValueError(`the k in mostSimilarItems >= 1`);
        let itemEmbeddingWeight = this.MFC.itemEmbeddingLayer.getWeights()[0];
        let similarity = euclideandistance(itemEmbeddingWeight, itemEmbeddingWeight.slice(id, 1))
        let { values, indices } = tf.topk(similarity, k + 1);
        let indicesArray = (indices.arraySync() as number[])
        return indicesArray.filter((e: number) => e !== id)
    }

    /**
       To save the architecture and the weights of the model in a given path
    */
    save(path: string): Promise<io.SaveResult> {
        return this.model.save('file://' + path);
    }

    /**
    * To load a pre-saved model 
    * 
    * if your data does not already have a DataBlock, only recommendItem method will work
    *   
    */
    async load(path: string): Promise<void> {
        this.model = await tf.loadLayersModel('file://' + path + '/model.json');
        this.usersNum = this.model.getWeights()[0].shape[0] -1;
        this.itemsNum = this.model.getWeights()[1].shape[0] -1;
        this.embeddingOutputSize = this.model.getWeights()[0].shape[1];
    }
}