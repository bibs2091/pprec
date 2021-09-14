import * as tf from '@tensorflow/tfjs-node'
import * as csv from 'fast-csv';
import * as fs from 'fs';

interface IdatasetInfo {
    size: number; usersNum: number; itemsNum: number;
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

interface optionsDataBlock {
    userColumn: string; itemColumn: string, ratingColumn: string; batchSize?: number;
    ratingRange?: number[]; validationPercentage?: number; delimiter?: string;
    seed?: number;
}
/**
    DataBlock is an api which allows you to generate and manupilate your dataset.
    To be used in the Learner API 
*/
export class DataBlock {
    trainingDataset: tf.data.Dataset<any>;
    validationDataset: tf.data.Dataset<any>;
    datasetInfo: IdatasetInfo;
    batchSize: number;
    ratingRange?: number[];


    /**
        Create a datablock from a csv file.
        You should define the name of the columns which contain the corresponding data 
    */
    async fromCsv(path: string, options: optionsDataBlock) {
        let myPath = "file://" + path;
        this.datasetInfo = await this.getInfoOnCsv(path, options.userColumn, options.itemColumn)
        this.ratingRange = options.ratingRange;
        let csvDataset: tf.data.Dataset<any> = (tf.data.csv(
            myPath, {
            configuredColumnsOnly: true,
            delimiter: options.delimiter,
            columnConfigs: {
                [options.userColumn]: {
                    required: true,
                    dtype: "float32"
                },
                [options.itemColumn]: {
                    required: true,
                    dtype: "float32"
                },
                [options.ratingColumn]: {
                    isLabel: true,
                    dtype: "float32"
                }
            }
        })).shuffle(
            (this.datasetInfo.size > 1e6) ? 1e6 : this.datasetInfo.size,
            (options?.seed == null) ? undefined : options?.seed.toString(),
            false
        ) //shuffle the dataset


        //split the dataset into train and valid set
        let validationPercentage = (options?.validationPercentage == null) ? 0.1 : options?.validationPercentage
        this.batchSize = (options?.batchSize == null) ? 64 : options?.batchSize

        let trainSize = Math.round((1 - validationPercentage) * this.datasetInfo.size)

        this.trainingDataset = csvDataset.take(trainSize).batch(this.batchSize);
        this.trainingDataset = this.trainingDataset.map((x: Idataset) => ({ xs: { user: x.xs[options.userColumn].reshape([-1, 1]), item: x.xs[options.itemColumn].reshape([-1, 1]) }, ys: x.ys }))

        if (validationPercentage > 0) {
            this.validationDataset = csvDataset.skip(trainSize).batch(this.batchSize);
            this.validationDataset = this.validationDataset.map((x: Idataset) => ({ xs: { user: x.xs[options.userColumn].reshape([-1, 1]), item: x.xs[options.itemColumn].reshape([-1, 1]) }, ys: x.ys }))

        }
        return this;
    }

    /**
        Create a datablock from a tensors.
        input the item, users, and ratings tensors
    */
    fromTensor(items: tf.Tensor, users: tf.Tensor, ratings: tf.Tensor, validationPercentage: number = 0, batchSize: number = 32, ratingRange: null | number[] = null, randomSeed: null | number[] = null, options: null | object = null): DataBlock {
        this.datasetInfo = { size: 0, usersNum: 0, itemsNum: 0 }
        this.datasetInfo.size = ratings.flatten().shape[0];

        // shuffle the dataset
        let randomTen = Array.from(tf.util.createShuffledIndices(this.datasetInfo.size));
        items = items.reshape([-1, 1]).gather(randomTen);
        users = users.reshape([-1, 1]).gather(randomTen);
        ratings = ratings.flatten().gather(randomTen);

        //train valid splitting
        if (validationPercentage > 0) {
            this.splitTrainValidTensor(items, users, ratings, validationPercentage)
        }
        else {
            let psuedoTrainingDataset: tf.TensorContainer[] = []
            for (let i = 0; i < ratings.shape[0]; i++) {
                psuedoTrainingDataset.push({ xs: { user: users.slice(i, 1), item: items.slice(i, 1) }, ys: { rating: ratings.slice(i) } })
            }
            this.trainingDataset = tf.data.array(psuedoTrainingDataset)
        }
        return this;
    }

    /**
        Get some stats about a csv file.
        mainly used in fromCsv method
        returns datasetInfo object
    */
    getInfoOnCsv(path: string, userColumn: string, itemColumn: string): Promise<IdatasetInfo> {
        let datasetInfo_ = new Promise<IdatasetInfo>(function (resolve, reject) {
            let csvInfo = { size: 0, usersNum: 0, itemsNum: 0 }
            let uniqueItems = new Set()
            let uniqueUsers = new Set()

            //using the fast-csv parse
            csv.parseFile(path, { headers: true })
                .on('error', error => console.error(error))
                .on('data', (data) => {
                    uniqueUsers.add(data[userColumn])
                    uniqueItems.add(data[itemColumn])
                })
                .on('end', (rowCount: number) => {
                    csvInfo.size = rowCount;
                    csvInfo.usersNum = uniqueUsers.size
                    csvInfo.itemsNum = uniqueItems.size
                    return resolve(csvInfo);
                })
        });
        return datasetInfo_;
    }

    getInfoOnCsv2(path: string, userColumn: string, itemColumn: string): Promise<IdatasetInfo> {
        let datasetInfo_ = new Promise<IdatasetInfo>(function (resolve, reject) {
            let csvInfo = { size: 0, usersNum: 0, itemsNum: 0 }
            let uniqueItems = new Set()
            let uniqueUsers = new Set()

            //using the fast-csv parse
            csv.parseFile(path, { headers: true })
                .on('error', error => console.error(error))
                .on('data', (data) => {
                    uniqueUsers.add(data[userColumn])
                    uniqueItems.add(data[itemColumn])
                })
                .on('end', (rowCount: number) => {
                    csvInfo.size = rowCount;
                    csvInfo.usersNum = uniqueUsers.size
                    csvInfo.itemsNum = uniqueItems.size
                    return resolve(csvInfo);
                })
        });
        return datasetInfo_;
    }


    /**
        Split the tensors into training and validation set.
        mainly used in fromTensor method
    */
    splitTrainValidTensor(items: tf.Tensor, users: tf.Tensor, ratings: tf.Tensor, validationPercentage: number): void {
        let trainSize: number = Math.round((1 - validationPercentage) * this.datasetInfo.size)
        let validSize: number = Math.abs(trainSize - this.datasetInfo.size)
        let [trainingItems, validationItems] = tf.split(items, [trainSize, validSize], 0);
        let [trainingUsers, validationUsers] = tf.split(users, [trainSize, validSize], 0);
        let [trainingRatings, validationRatings] = tf.split(ratings, [trainSize, validSize], 0);

        let psuedoTrainingDataset: tf.TensorContainer[] = []
        for (let i = 0; i < trainingRatings.shape[0]; i++) {
            psuedoTrainingDataset.push({ xs: { user: trainingUsers.slice(i, 1), item: trainingItems.slice(i, 1) }, ys: { rating: trainingRatings.slice(i) } })
        }
        this.trainingDataset = tf.data.array((psuedoTrainingDataset))


        let psuedoValidationDataset: tf.TensorContainer[] = []
        for (let i = 0; i < validationRatings.shape[0]; i++) {
            psuedoValidationDataset.push({ xs: { user: validationUsers.slice(i, 1), item: validationItems.slice(i, 1) }, ys: { rating: validationRatings.slice(i) } })
        }
        this.validationDataset = tf.data.array((psuedoValidationDataset))
    }


    /**
        save the datablock in a path (training + validation)
    */
    async save(outputFile: string): Promise<void> {
        console.time("dbsave");
        const writeStream = fs.createWriteStream(outputFile);
        const stream = csv.format({ headers: ['user', 'item', 'rating'] });
        await this.trainingDataset.forEachAsync(
            function (e: Idataset) {
                let users_ = e.xs.user.dataSync()
                let items_ = e.xs.item.dataSync()
                let ratings_ = e.ys.rating.dataSync()
                for (let i = 0; i < ratings_.length; i++)
                    stream.write([users_[i], items_[i], ratings_[i]])
            }
        );

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
}
