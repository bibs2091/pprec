import { MatrixFactorization } from './MatrixFactorization';
import * as tf from '@tensorflow/tfjs-node'
import { DataBlock } from './DataBlock'
import { cosineSimilarity, euclideandistance } from './utils'
import { ValueError, NonExistance } from './errors'
import { io, range } from '@tensorflow/tfjs-core';
import * as fs from 'fs';


interface optionsLearner {
    learningRate?: number; embeddingOutputSize?: number;
    lossFunc?: string; optimizerName?: string;
    l2Labmda?: number; redisUrl?: string
}

/**
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
    MFC?: MatrixFactorization;
    ratingRange?: number[];
    optimizerName: string;
    dataBlock: DataBlock;
    l2Labmda?: number;
    modelToUserMap: Map<number, any>;
    modelToItemMap: Map<number, any>;
    constructor(dataBlock?: DataBlock, options?: optionsLearner) {
        if (dataBlock != null) {
            this.dataBlock = dataBlock;
            this.itemsNum = this.dataBlock.datasetInfo.itemsNum;
            this.usersNum = this.dataBlock.datasetInfo.usersNum;
            this.ratingRange = this.dataBlock.ratingRange
            this.learningRate = options?.learningRate ? options?.learningRate : 1e-3 ;


            if (options?.lossFunc == null) this.lossFunc = "meanSquaredError";
            else this.lossFunc = options?.lossFunc;

            if (options?.embeddingOutputSize == null) this.embeddingOutputSize = 3;
            else this.embeddingOutputSize = options?.embeddingOutputSize;

            if (options?.l2Labmda == null) this.l2Labmda = 0;
            else this.l2Labmda = options?.l2Labmda;

            this.MFC = new MatrixFactorization(this.usersNum, this.itemsNum, this.embeddingOutputSize, this.l2Labmda, this.ratingRange);
            this.model = this.MFC.model;

            if (options?.optimizerName == null) this.optimizerName = "adam";
            else this.optimizerName = options?.optimizerName;
            this.setOptimizer(this.optimizerName);
            this.modelToUserMap = new Map();
            this.modelToItemMap = new Map();
            this.dataBlock.datasetInfo.userToModelMap.forEach((value, key) => this.modelToUserMap.set(value, key));
            this.dataBlock.datasetInfo.itemToModelMap.forEach((value, key) => this.modelToItemMap.set(value, key));
        }
        else {
            this.lossFunc = "meanSquaredError";
            this.optimizerName = "adam";
            this.dataBlock = new DataBlock(options?.redisUrl)
            this.itemsNum = this.dataBlock.datasetInfo.itemsNum;
            this.usersNum = this.dataBlock.datasetInfo.usersNum;
            if (options?.lossFunc == null) this.lossFunc = "meanSquaredError";
            else this.lossFunc = options.lossFunc;

            if (options?.embeddingOutputSize == null) this.embeddingOutputSize = 3;
            else this.embeddingOutputSize = options?.embeddingOutputSize;

            if (options?.l2Labmda == null) this.l2Labmda = 0;
            else this.l2Labmda = options?.l2Labmda;
            this.MFC = new MatrixFactorization(1, 1, this.embeddingOutputSize, this.l2Labmda, this.ratingRange);
            this.model = this.MFC.model;

            this.dataBlock.datasetInfo.userToModelMap.set('0', 0)
            this.dataBlock.datasetInfo.itemToModelMap.set('0', 0)
            this.modelToUserMap = new Map();
            this.modelToItemMap = new Map();
            this.dataBlock.datasetInfo.userToModelMap.forEach((value, key) => this.modelToUserMap.set(value, key));
            this.dataBlock.datasetInfo.itemToModelMap.forEach((value, key) => this.modelToItemMap.set(value, key));
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

        if (this.dataBlock.validationDataset && this.dataBlock.validationDataset.size > 0)
            return this.model.fitDataset(this.dataBlock.trainingDataset, {
                validationData: this.dataBlock.validationDataset,
                epochs: epochs,
            })
        else
            return this.model.fitDataset(this.dataBlock.trainingDataset, {
                epochs: epochs,
            })
    }

    /**
        To recommend k items for a user given their ID
    */
    async recommendItems(userId: any, k: number, alreadyWatched: boolean = false): Promise<number[]> {

        if (k > this.itemsNum)
            throw new ValueError(`The value k = ${k} is bigger than the number of items you actually have.`);


        let userIdMapped = this.dataBlock.datasetInfo.userToModelMap.get(`${userId}`)
        
        if (userIdMapped == null)
            throw new NonExistance(`The user ${userId} does not exist, please recheck the ID again.`);


        // to fix map
        let toPredict = [
            tf.fill([this.itemsNum, 1], userIdMapped),
            tf.range(0, this.itemsNum).reshape([-1, 1])];


        let toPredictResults = (this.model.predictOnBatch(toPredict) as tf.Tensor).flatten()

        if (this.dataBlock && !alreadyWatched) {
            let PredictMask = new Array(toPredictResults.shape[0]).fill(true);
            let usersIdMovies = await this.dataBlock.client.SMEMBERS(userIdMapped.toString())
            PredictMask = PredictMask.map(function (value, index) {
                return usersIdMovies.indexOf(index.toString()) == -1;
            })
            toPredictResults = toPredictResults.where(PredictMask, [-100]);

        }
        const { values, indices } = tf.topk(toPredictResults, k);
        return (indices.arraySync() as number[]).map(e => this.modelToItemMap.get(e));
    }

    /**
        To add a rating of a user on a certain item and add it to redis database
    */
    addRating(userId: any, itemId: any, rating: any, train: boolean = true): void | Promise<tf.History> {

        let userIdMapped = (this.dataBlock.datasetInfo.userToModelMap.get(`${userId}`) as number);
        let itemIdMapped = (this.dataBlock.datasetInfo.itemToModelMap.get(`${itemId}`) as number);

        if (userIdMapped == null)
            throw new NonExistance(`The user ${userId} does not exist, please recheck the ID again.`);

        if (itemIdMapped == null)
            throw new NonExistance(`The item ${itemId} does not exist, please recheck the ID again.`);

        let toAdd = tf.data.array([
            {
                xs: {
                    // map the user id and item id to the model index
                    user: tf.tensor2d([[userIdMapped]]),
                    item: tf.tensor2d([[itemIdMapped]])
                },
                ys: { rating: tf.tensor1d([Number(rating)]) }
            }])
        this.dataBlock.client.SADD(userIdMapped.toString(), itemIdMapped.toString());
        if (this.dataBlock.datasetInfo.size == 0) {
            this.dataBlock.trainingDataset = toAdd
            this.dataBlock.datasetInfo.size++;
        }
        else {
            this.dataBlock.trainingDataset = this.dataBlock.trainingDataset.concatenate(toAdd);
            this.dataBlock.datasetInfo.size++;
        }

        if (train) {
            return this.model.fitDataset(toAdd, {
                epochs: 1,
                verbose: 0
            })
        }
    }


    async addRatingSync(userId: any, itemId: any, rating: any, train: boolean = true) {

        let userIdMapped = (this.dataBlock.datasetInfo.userToModelMap.get(`${userId}`) as number);
        let itemIdMapped = (this.dataBlock.datasetInfo.itemToModelMap.get(`${itemId}`) as number);


        if (userIdMapped == null)
            throw new NonExistance(`The user ${userId} does not exist, please recheck the ID again.`);

        if (itemIdMapped == null)
            throw new NonExistance(`The item ${itemId} does not exist, please recheck the ID again.`);


        let toAdd = tf.data.array([
            {
                xs: {
                    // map the user id and item id to the model index
                    user: tf.tensor2d([[userIdMapped]]),
                    item: tf.tensor2d([[itemIdMapped]])
                },
                ys: { rating: tf.tensor1d([Number(rating)]) }
            }])

        await this.dataBlock.client.SADD(userIdMapped.toString(), itemIdMapped.toString());
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

        this.usersNum += 1
        this.dataBlock.datasetInfo.usersNum += 1
        this.dataBlock.datasetInfo.userToModelMap.set(`${userId}`, this.usersNum - 1);
        this.modelToUserMap.set(this.usersNum - 1, `${userId}`);

        //check if datablock existed, because of userNums == 1 this means datablock didn't exist.
        if (this.usersNum > 1) {
            let userEmbeddingWeight = this.model.getWeights()[0];
            userEmbeddingWeight = tf.concat([userEmbeddingWeight, userEmbeddingWeight.mean(0).reshape([1, this.embeddingOutputSize])]);
            this.MFC = new MatrixFactorization(this.usersNum, this.itemsNum == 0 ? 1 : this.itemsNum, this.embeddingOutputSize, 0, this.ratingRange, [userEmbeddingWeight]);
        }
        else {
            this.MFC = new MatrixFactorization(this.usersNum, this.itemsNum == 0 ? 1 : this.itemsNum, this.embeddingOutputSize, 0, this.ratingRange);
        }
        this.model = this.MFC.model;
        this.setOptimizer(this.optimizerName);
    }

    /**
        To add a new item embedding in the model.
        The embedding is generated based on the mean of the other item latent factors.
    */
    newItem(itemId: any) {

        if (this.embeddingOutputSize == null)
            throw new NonExistance(`embeddingOutputSize does not exist`);

        this.itemsNum += 1;
        this.dataBlock.datasetInfo.itemsNum += 1
        this.dataBlock.datasetInfo.itemToModelMap.set(`${itemId}`, this.itemsNum - 1);
        this.modelToItemMap.set(this.itemsNum - 1, `${itemId}`);

        if (this.itemsNum > 1) {
            let itemEmbeddingWeight = this.model.getWeights()[1];
            itemEmbeddingWeight = tf.concat([itemEmbeddingWeight, itemEmbeddingWeight.mean(0).reshape([1, this.embeddingOutputSize])]);
            this.MFC = new MatrixFactorization(this.usersNum == 0 ? 1 : this.usersNum, this.itemsNum, this.embeddingOutputSize, 0, this.ratingRange, undefined, [itemEmbeddingWeight]);
        }
        else {
            this.MFC = new MatrixFactorization(this.usersNum == 0 ? 1 : this.usersNum, this.itemsNum, this.embeddingOutputSize, 0, this.ratingRange, undefined);
        }
        this.model = this.MFC.model;
        this.setOptimizer(this.optimizerName);
    }



    /**
       To retrieve the k similar users of a user 
    */
    mostSimilarUsers(id: any, k = 10): string[] {

        if (k < 1) throw new ValueError(`the k in mostSimilarUsers >= 1`);
        if (k > this.usersNum)
            throw new ValueError(`The value k = ${k} is bigger than the number of users you actually have.`);

        let userEmbeddingWeight = this.model.getWeights()[0];
        let mappedId = this.dataBlock.datasetInfo.userToModelMap.get(`${id}`) as number

        if (mappedId == null)
            throw new NonExistance(`The user ${id} does not exist, please recheck the ID again.`);


        let similarity = cosineSimilarity(userEmbeddingWeight, userEmbeddingWeight.slice(mappedId, 1));
        let { values, indices } = tf.topk(similarity, k + 1);
        let indicesArray = (indices.arraySync() as number[])
        return indicesArray.filter((e: number) => e !== mappedId).map(e => this.modelToUserMap.get(e));
    }

    /**
       To retrieve the k similar items of an item 
    */
    mostSimilarItems(id: any, k = 10): number[] {

        if (k < 1) throw new ValueError(`the k in mostSimilarItems >= 1`);
        if (k > this.itemsNum)
            throw new ValueError(`The value k = ${k} is bigger than the number of items you actually have.`);

        let itemEmbeddingWeight = this.model.getWeights()[1];
        let mappedId = this.dataBlock.datasetInfo.itemToModelMap.get(`${id}`)

        if (mappedId == null)
            throw new NonExistance(`The item ${id} does not exist, please recheck the ID again.`);



        let similarity = cosineSimilarity(itemEmbeddingWeight, itemEmbeddingWeight.slice(mappedId, 1))
        let { values, indices } = tf.topk(similarity, k + 1);
        let indicesArray = (indices.arraySync() as number[])
        return indicesArray.filter((e: number) => e !== mappedId).map(e => this.modelToItemMap.get(e));
    }

    /**
      Use this when a user view an item but did not rate it, allowing pprec to not re-recommend this item
    */
    viewed(userId: any, itemId: any) {
        let userIdMapped = (this.dataBlock.datasetInfo.userToModelMap.get(`${userId}`) as number);
        let itemIdMapped = (this.dataBlock.datasetInfo.itemToModelMap.get(`${itemId}`) as number);
        this.dataBlock.client.SADD(userIdMapped.toString(), itemIdMapped.toString());
    }

    /**
       To save the architecture and the weights and id Maps of the model in a given path
    */
    save(path: string): Promise<io.SaveResult> {

        // in case the folder does not already exist: create it
        if (!fs.existsSync(path)) {
            fs.mkdirSync(path);
        }
        let userMap = JSON.stringify(Array.from((this.dataBlock.datasetInfo.userToModelMap as Map<any, number>).entries()))
        fs.writeFileSync(`${path}/userToModelMap.txt`, userMap);

        let itemMap = JSON.stringify(Array.from((this.dataBlock.datasetInfo.itemToModelMap as Map<any, number>).entries()))
        fs.writeFileSync(`${path}/itemToModelMap.txt`, itemMap);

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
        this.embeddingOutputSize = this.model.getWeights()[0].shape[1] as number;
        this.setOptimizer(this.optimizerName);

        // load userToModelMap and itemToModelMap
        let loadedUserMap: string = await fs.readFileSync(`${path}/userToModelMap.txt`, 'utf8');
        let loadedItemMap: string = await fs.readFileSync(`${path}/itemToModelMap.txt`, 'utf8');

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