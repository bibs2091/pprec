import { MatrixFactorization } from './MatrixFactorization';
import * as tf from '@tensorflow/tfjs-node'
import { DataBlock } from './DataBlock'
import { cosineSimilarity, euclideandistance } from './utils'
import { ValueError, NonExistance } from './errors'
import { io } from '@tensorflow/tfjs-core';
import * as fs from 'fs';

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
    modelToUserMap: Map<number, any>;
    modelToItemMap: Map<number, any>;
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
            this.modelToUserMap = new Map();
            this.modelToItemMap = new Map();
            this.dataBlock.datasetInfo.userToModelMap.forEach((value, key) => this.modelToUserMap.set(value, key));
            this.dataBlock.datasetInfo.itemToModelMap.forEach((value, key) => this.modelToItemMap.set(value, key));

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
    recommendItems(userId: number, k: number): number[] {
        if (this.itemsNum == null)
            throw new NonExistance(
                `itemsNum does not exist, this is maybe because you did not feed Learner a DataBlock`
            );

        if (this.model == null)
            throw new NonExistance(`No model to train, please provoid a proper model`);

        // to fix map
        let toPredict = [
            tf.fill([this.itemsNum, 1], (this.dataBlock?.datasetInfo.userToModelMap.get(`${userId}`) as number)),
            tf.range(0, this.itemsNum).reshape([-1, 1])];
        const { values, indices } = tf.topk((this.model.predictOnBatch(toPredict) as tf.Tensor).flatten(), k);

        return (indices.arraySync() as number[]).map(e => this.modelToItemMap.get(e));
    }

    addRating(userId: any, itemId: any, rating: any, train: boolean = true): void | Promise<tf.History> {

        if (this.dataBlock == null)
            throw new NonExistance(`No datablock to train on, please provoid a proper DataBlock `);

        let toAdd = tf.data.array([
            {
                xs: {
                    // map the user id and item id to the model index
                    user: tf.tensor2d([[(this.dataBlock.datasetInfo.userToModelMap.get(`${userId}`) as number)]]),
                    item: tf.tensor2d([[(this.dataBlock.datasetInfo.itemToModelMap.get(`${itemId}`) as number)]])
                },
                ys: { rating: tf.tensor1d([Number(rating)]) }
            }])

        if (this.dataBlock.datasetInfo.size == 0) {
            this.dataBlock.trainingDataset = toAdd
            this.dataBlock.datasetInfo.size++;
        }
        else {
            this.dataBlock.trainingDataset = this.dataBlock.trainingDataset.concatenate(toAdd);
            this.dataBlock.datasetInfo.size++;
        }

        if (train) {
            if (this.model == null)
                throw new NonExistance(`No model to train, please provoid a proper model`);

            return this.model.fitDataset(toAdd, {
                epochs: 1,
                verbose: 0
            })
        }
    }


    async addRatingSync(userId: any, itemId: any, rating: any, train: boolean = true) {

        if (this.dataBlock == null)
            throw new NonExistance(`No datablock to train on, please provoid a proper DataBlock `);

        let toAdd = tf.data.array([
            {
                xs: {
                    // map the user id and item id to the model index
                    user: tf.tensor2d([[(this.dataBlock.datasetInfo.userToModelMap.get(`${userId}`) as number)]]),
                    item: tf.tensor2d([[(this.dataBlock.datasetInfo.itemToModelMap.get(`${itemId}`) as number)]])
                },
                ys: { rating: tf.tensor1d([Number(rating)]) }
            }])

        if (this.dataBlock.datasetInfo.size == 0) {
            this.dataBlock.trainingDataset = toAdd
            this.dataBlock.datasetInfo.size++;
        }
        else {
            this.dataBlock.trainingDataset = this.dataBlock.trainingDataset.concatenate(toAdd);
            this.dataBlock.datasetInfo.size++;
        }

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
    newUser(userId: any) {
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
        this.dataBlock?.datasetInfo.userToModelMap.set(userId, this.usersNum);
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
    newItem(itemId: any) {
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

        this.itemsNum += 1;
        this.dataBlock?.datasetInfo.itemToModelMap.set(itemId, this.itemsNum);
        let itemEmbeddingWeight = this.MFC.itemEmbeddingLayer.getWeights()[0];
        itemEmbeddingWeight = tf.concat([itemEmbeddingWeight, itemEmbeddingWeight.mean(0).reshape([1, this.embeddingOutputSize])]);
        this.MFC = new MatrixFactorization(this.usersNum, this.itemsNum, this.embeddingOutputSize, 0, this.ratingRange, undefined, [itemEmbeddingWeight]);
        this.model = this.MFC.model;
        this.setOptimizer(this.optimizerName);
        return this.itemsNum //the new user ID
    }



    /**
       To retrieve the k similar users of a user 
    */
    mostSimilarUsers(id: any, k = 10): string[] {

        if (this.model == null || this.MFC == null)
            throw new NonExistance(`No model to train, please provoid a proper model`);

        if (k < 1) throw new ValueError(`the k in mostSimilarUsers >= 1`);

        let userEmbeddingWeight = this.MFC.userEmbeddingLayer.getWeights()[0];
        let mappedId = this.dataBlock?.datasetInfo.userToModelMap.get(`${id}`) as number
        let similarity = cosineSimilarity(userEmbeddingWeight, userEmbeddingWeight.slice(mappedId, 1));
        let { values, indices } = tf.topk(similarity, k + 1);
        let indicesArray = (indices.arraySync() as number[])
        return indicesArray.filter((e: number) => e !== mappedId).map(e => this.modelToUserMap.get(e));
    }

    /**
       To retrieve the k similar items of an item 
    */
    mostSimilarItems(id: any, k = 10): number[] {
        if (this.model == null || this.MFC == null)
            throw new NonExistance(`No model to train, please provoid a proper model`);

        if (k < 1) throw new ValueError(`the k in mostSimilarItems >= 1`);

        let itemEmbeddingWeight = this.MFC.itemEmbeddingLayer.getWeights()[0];
        let mappedId = this.dataBlock?.datasetInfo.itemToModelMap.get(`${id}`) as number
        let similarity = euclideandistance(itemEmbeddingWeight, itemEmbeddingWeight.slice(mappedId, 1))
        let { values, indices } = tf.topk(similarity, k + 1);
        let indicesArray = (indices.arraySync() as number[])
        return indicesArray.filter((e: number) => e !== mappedId).map(e => this.modelToItemMap.get(e));
    }

    /**
       To save the architecture and the weights and id Maps of the model in a given path
    */
    save(path: string): Promise<io.SaveResult> {
        let userMap = JSON.stringify(Array.from((this.dataBlock?.datasetInfo.userToModelMap as Map<any, number>).entries()))
        fs.writeFileSync(`${path}_userToModelMap.txt`, userMap)

        let itemMap = JSON.stringify(Array.from((this.dataBlock?.datasetInfo.itemToModelMap as Map<any, number>).entries()))
        fs.writeFileSync(`${path}_itemToModelMap.txt`, itemMap)


        return this.model.save('file://' + path);
    }

    /**
    * To load a pre-saved model
    * 
    * if your data does not already have a DataBlock, only recommendItem method will work
    *   
    */
    async load(path: string): Promise<Learner> {





        this.model = await tf.loadLayersModel('file://' + path + '/model.json');
        this.usersNum = this.model.getWeights()[0].shape[0] - 1;
        this.itemsNum = this.model.getWeights()[1].shape[0] - 1;
        this.embeddingOutputSize = this.model.getWeights()[0].shape[1];
        this.setOptimizer(this.optimizerName);

        // load userToModelMap and itemToModelMap
        let loadedUserMap: string = await fs.readFileSync(`${path}_userToModelMap.txt`, 'utf8');
        let loadedItemMap: string = await fs.readFileSync(`${path}_itemToModelMap.txt`, 'utf8');

        if (this.dataBlock == null) {
            this.dataBlock = new DataBlock()
            this.modelToUserMap = new Map();
            this.modelToItemMap = new Map();
        }

        this.dataBlock.datasetInfo = {
            size: this.dataBlock.datasetInfo == null ? 0 : this.dataBlock.datasetInfo.size,
            usersNum: this.usersNum, itemsNum: this.itemsNum,
            userToModelMap: new Map(JSON.parse(loadedUserMap)), itemToModelMap: new Map(JSON.parse(loadedItemMap))
        }
        this.dataBlock.datasetInfo.userToModelMap.forEach((value, key) => this.modelToUserMap.set(value, key));
        this.dataBlock.datasetInfo.itemToModelMap.forEach((value, key) => this.modelToItemMap.set(value, key));


        return this;
    }
}