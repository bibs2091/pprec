import * as tf from '@tensorflow/tfjs-node'
import * as csv from 'fast-csv';
import * as fs from 'fs';
import { createClient } from 'redis';
interface IdatasetInfo {
    size: number; usersNum: number; itemsNum: number;
    userToModelMap: Map<any, number>, itemToModelMap: Map<any, number>
}

interface Idataset {
    xs: {
        user: tf.Tensor;
        item: tf.Tensor;
    }
    ys: {
        rating: tf.Tensor;
    }
}

interface Idataset2 {
    xs: {
        [user_item: string]: any;
    }
    ys: {
        rating: any;
    }
}


interface optionsDataBlockCsv {
    userColumn: string; itemColumn: string, ratingColumn: string; batchSize?: number;
    ratingRange?: number[]; validationPercentage?: number; delimiter?: string;
    seed?: number;
}

interface optionsDataBlockArray {
    batchSize?: number; ratingRange?: number[]; 
    validationPercentage?: number; seed?: number;
}
/**
    DataBlock is an api which allows you to generate and manupilate your dataset.
    To be used in the Learner API 
*/
export class DataBlock {
    trainingDataset: tf.data.Dataset<any>;
    validationDataset: tf.data.Dataset<any>;
    datasetInfo: IdatasetInfo;
    usersMovies: any;
    batchSize: number;
    client: any;
    ratingRange?: number[];


    constructor(redisUrl?: string) {
        this.datasetInfo = { size: 0, usersNum: 0, itemsNum: 0, userToModelMap: new Map(), itemToModelMap: new Map() }
        this.redisConfig(redisUrl).then(e => console.log("connected"))
    }

    /**
        Create a datablock from a csv file.
        You should define the name of the columns which contain the corresponding data 
    */
    async fromCsv(path: string, options: optionsDataBlockCsv) {
        let myPath = "file://" + path;

        options.delimiter = (options.delimiter == null) ? ',' : options?.delimiter;
        this.datasetInfo = await this.getInfoOnCsv(path, options.userColumn, options.itemColumn, options.delimiter)
        this.ratingRange = options.ratingRange;
        let csvDataset: tf.data.Dataset<any> = (tf.data.csv(
            myPath, {
            configuredColumnsOnly: true,
            delimiter: options.delimiter,
            columnConfigs: {
                [options.userColumn]: {
                    required: true,
                    // dtype: "float32"
                },
                [options.itemColumn]: {
                    required: true,
                    // dtype: "float32"
                },
                [options.ratingColumn]: {
                    isLabel: true,
                    // dtype: "float32"
                }
            }
        })).shuffle(  //shuffle the dataset
            (this.datasetInfo.size > 1e5) ? 1e5 : this.datasetInfo.size,
            (options?.seed == null) ? undefined : options?.seed.toString(),
            false
        ).map((x: any) => (
            { xs: { user: this.datasetInfo.userToModelMap.get(`${x.xs[options.userColumn]}`), item: this.datasetInfo.itemToModelMap.get(`${x.xs[options.itemColumn]}`) }, ys: { rating: Number(x.ys[options.ratingColumn]) } }
        ))


        //split the dataset into train and valid set
        let validationPercentage = (options?.validationPercentage == null) ? 0.1 : options?.validationPercentage
        this.batchSize = (options?.batchSize == null) ? 64 : options?.batchSize

        let trainSize = Math.round((1 - validationPercentage) * this.datasetInfo.size)

        this.trainingDataset = csvDataset.take(trainSize).batch(this.batchSize);
        this.trainingDataset = this.trainingDataset.map((x: Idataset) => ({ xs: { user: x.xs.user.reshape([-1, 1]), item: x.xs.item.reshape([-1, 1]) }, ys: { rating: x.ys.rating.reshape([-1, 1]) } }))

        if (validationPercentage > 0) {
            this.validationDataset = csvDataset.skip(trainSize)
                .batch(this.batchSize);
            this.validationDataset = this.validationDataset.map((x: Idataset) => ({ xs: { user: x.xs.user.reshape([-1, 1]), item: x.xs.item.reshape([-1, 1]) }, ys: { rating: x.ys.rating.reshape([-1, 1]) } }))
        }
        return this;
    }

    /**
        Create a datablock from a tensors.
        input the item, users, and ratings tensors
    */
    async fromArray(items: number[], users: number[], ratings: number[], options?: optionsDataBlockArray) {
        this.datasetInfo.itemsNum = new Set(items).size;
        this.datasetInfo.usersNum = new Set(users).size;
        this.datasetInfo.size = ratings.length;
        this.batchSize = (options?.batchSize == null) ? 64 : options?.batchSize
        let validationPercentage = options?.validationPercentage ? options?.validationPercentage : 0.1 
        
        
        // shuffle the dataset
        let randomTen = Array.from(tf.util.createShuffledIndices(this.datasetInfo.size));
        items = randomTen.map(i => items[i]);
        users = randomTen.map(i => users[i]);
        ratings = randomTen.map(i => ratings[i]);


        //train valid splitting
        if (validationPercentage > 0) {
            this.splitTrainValidTensor(items, users, ratings, validationPercentage)
        }
        else {
            let psuedoTrainingDataset: tf.TensorContainer[] = []
            for (let i = 0; i < items.length; i++) {
                psuedoTrainingDataset.push({ xs: { user: users[i], item: items[i] }, ys: { rating: ratings[i] } })
            }
            this.trainingDataset = tf.data.array(psuedoTrainingDataset)
        }

        this.trainingDataset = this.trainingDataset.batch(this.batchSize);
        this.validationDataset = this.validationDataset.batch(this.batchSize);

        this.trainingDataset = this.trainingDataset.map((x: Idataset) => ({ xs: { user: x.xs.user.reshape([-1, 1]), item: x.xs.item.reshape([-1, 1]) }, ys: { rating: x.ys.rating.reshape([-1, 1]) } }))
        this.validationDataset = this.validationDataset.map((x: Idataset) => ({ xs: { user: x.xs.user.reshape([-1, 1]), item: x.xs.item.reshape([-1, 1]) }, ys: { rating: x.ys.rating.reshape([-1, 1]) } }))

        return this;
    }

    /**
        Get some stats about a csv file.
        mainly used in fromCsv method
        returns datasetInfo object
    */
    getInfoOnCsv(path: string, userColumn: string, itemColumn: string, delimiter: string): Promise<IdatasetInfo> {
        let client = this.client
        let datasetInfo_ = new Promise<IdatasetInfo>(function (resolve, reject) {
            let csvInfo = { size: 0, usersNum: 0, itemsNum: 0, userToModelMap: new Map(), itemToModelMap: new Map() };
            let usersIndex: number = 0;
            let itemsIndex: number = 0;
            //using the fast-csv parse
            csv.parseFile(path, { headers: true, delimiter: delimiter })
                .on('error', error => console.error(error))
                .on('data', (data) => {
                    if (!csvInfo.userToModelMap.has(`${data[userColumn]}`)) {
                        csvInfo.userToModelMap.set(`${data[userColumn]}`, usersIndex);
                        usersIndex += 1;
                    }
                    if (!csvInfo.itemToModelMap.has(`${data[itemColumn]}`)) {
                        csvInfo.itemToModelMap.set(`${data[itemColumn]}`, itemsIndex);
                        itemsIndex += 1;
                    }

                    client.SADD(
                        csvInfo.userToModelMap.get(`${data[userColumn]}`).toString(),
                        csvInfo.itemToModelMap.get(`${data[itemColumn]}`).toString()
                    );

                })
                .on('end', (rowCount: number) => {
                    csvInfo.size = rowCount;
                    csvInfo.usersNum = usersIndex
                    csvInfo.itemsNum = itemsIndex
                    return resolve(csvInfo);
                })
        });
        return datasetInfo_;
    }



    /**
        Split the tensors into training and validation set.
        mainly used in fromTensor method
    */
    splitTrainValidTensor(items: number[], users: number[], ratings: number[], validationPercentage: number): void {
        let trainSize: number = Math.round((1 - validationPercentage) * this.datasetInfo.size);
        let usersIndex = 0;
        let itemsIndex = 0;
        // splitting
        let trainingItems = items.slice(0, trainSize);
        let validationItems = items.slice(trainSize);

        let trainingUsers = users.slice(0, trainSize);
        let validationUsers = users.slice(trainSize);

        let trainingRatings = ratings.slice(0, trainSize)
        let validationRatings = ratings.slice(trainSize)

        let psuedoTrainingDataset: tf.TensorContainer[] = []
        for (let i = 0; i < trainingRatings.length; i++) {
            psuedoTrainingDataset.push({ xs: { user: trainingUsers[i], item: trainingItems[i] }, ys: { rating: trainingRatings[i] } })
            
            if (!this.datasetInfo.userToModelMap.has(`${trainingUsers[i]}`)) {
                this.datasetInfo.userToModelMap.set(`${trainingUsers[i]}`, usersIndex);
                usersIndex += 1;
            }

            if (!this.datasetInfo.itemToModelMap.has(`${trainingItems[i]}`)) {
                this.datasetInfo.itemToModelMap.set(`${trainingItems[i]}`, itemsIndex);
                itemsIndex += 1;
            }

            this.client.SADD(
                (this.datasetInfo.userToModelMap.get(`${trainingUsers[i]}`) as number).toString(),
                (this.datasetInfo.itemToModelMap.get(`${trainingItems[i]}`) as number).toString()
            );

        }
        this.trainingDataset = tf.data.array((psuedoTrainingDataset))


        let psuedoValidationDataset: tf.TensorContainer[] = []
        for (let i = 0; i < validationRatings.length; i++) {
            psuedoValidationDataset.push({ xs: { user: validationUsers[i], item: validationItems[i] }, ys: { rating: validationRatings[i] } })
            
            if (!this.datasetInfo.userToModelMap.has(`${validationUsers[i]}`)) {
                this.datasetInfo.userToModelMap.set(`${validationUsers[i]}`, usersIndex);
                usersIndex += 1;
            }

            if (!this.datasetInfo.itemToModelMap.has(`${validationItems[i]}`)) {
                this.datasetInfo.itemToModelMap.set(`${validationItems[i]}`, itemsIndex);
                itemsIndex += 1;
            }

            this.client.SADD(
                (this.datasetInfo.userToModelMap.get(`${validationUsers[i]}`) as number).toString(),
                (this.datasetInfo.itemToModelMap.get(`${validationItems[i]}`) as number).toString()
            );
        }

        this.validationDataset = tf.data.array((psuedoValidationDataset))
    }

/**
        save the datablock in a path (training + validation).
        In case you wanted to save the validation data in different file, write the validation file name in the second argument "validationFileName"
    */
    async save(outputFile: string, validationFileName?: string): Promise<void> {
        let writeStream = fs.createWriteStream(outputFile);
        let stream = csv.format({ headers: ['user', 'item', 'rating'] });
        await this.trainingDataset.forEachAsync(
            function (e: Idataset) {
                let users_ = e.xs.user.dataSync()
                let items_ = e.xs.item.dataSync()
                let ratings_ = e.ys.rating.dataSync()
                for (let i = 0; i < ratings_.length; i++)
                    stream.write([users_[i], items_[i], ratings_[i]])
            }
        );

        if (validationFileName != null) {
            console.log("validation");
            
            stream.end();
            stream.pipe(writeStream);
            writeStream = fs.createWriteStream(validationFileName);
            stream = csv.format({ headers: ['user', 'item', 'rating'] });
        }

        if (this.validationDataset != null)
            await this.validationDataset.forEachAsync(
                function (e: Idataset) {
                    let users_ = e.xs.user.dataSync()
                    let items_ = e.xs.item.dataSync()
                    let ratings_ = e.ys.rating.dataSync()
                    for (let i = 0; i < ratings_.length; i++)
                        stream.write([users_[i], items_[i], ratings_[i]])
                }
            );

        stream.end();
        stream.pipe(writeStream);
    }


    /**
        return the size of the dataset (training + validation)
    */
    size(): number {
        return this.datasetInfo.size;
    }


    async redisConfig(url) {
        if (url == null)
            this.client = createClient();
        else
            this.client = createClient({
                url: url
            });
        this.client.on('error', (err) => console.log('Redis Client Error', err));
        await this.client.connect()
    }

}
