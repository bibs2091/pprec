import * as tf from '@tensorflow/tfjs-node'
import * as csv from '@fast-csv/parse';

interface IdatasetInfo {
    size: number; usersNum: number; itemsNum: number;
}

export class DataBlock {
    trainingDataset: tf.data.Dataset<any>;
    validationDataset: tf.data.Dataset<any>;
    datasetInfo: IdatasetInfo;
    ratingRange: null | number[];
    async fromCsv(path: string, userColumn: string, itemColumn: string, ratingColumn: string, validationPercentage: number = 0.2, delimiter: string = ',', batchSize: number = 16, ratingRange: null | number[] = null, seed: number = 42, options: null | object = null) {
        let myPath = "file://" + path;
        this.datasetInfo = await this.getInfoOnCsv(path, userColumn, itemColumn)
        this.ratingRange = ratingRange;


        let csvDataset: tf.data.Dataset<any> = (tf.data.csv(
            myPath, {
            configuredColumnsOnly: true,
            delimiter: delimiter,
            // hasHeader: header,
            columnConfigs: {
                [userColumn]: {
                    required: true,
                    dtype: "float32"
                },
                [itemColumn]: {
                    required: true,
                    dtype: "float32"
                },
                [ratingColumn]: {
                    isLabel: true,
                    dtype: "float32"
                }
            }
        })).shuffle(this.datasetInfo.size, seed.toString(), false)


        let trainSize = Math.round((1 - validationPercentage) * this.datasetInfo.size)

        this.trainingDataset = csvDataset.take(trainSize).batch(batchSize);
        this.trainingDataset = this.trainingDataset.map(x => ({ xs: { user: x.xs[userColumn].reshape([-1, 1]), item: x.xs[itemColumn].reshape([-1, 1]) }, ys: x.ys }))

        this.validationDataset = csvDataset.skip(trainSize).batch(batchSize);
        this.validationDataset = this.validationDataset.map(x => ({ xs: { user: x.xs[userColumn].reshape([-1, 1]), item: x.xs[itemColumn].reshape([-1, 1]) }, ys: x.ys }))
    }

    fromTensor(items: tf.Tensor, users: tf.Tensor, ratings: tf.Tensor, validationPercentage: number = 0, batchSize: number = 32, ratingRange: null | number[] = null, randomSeed: null | number[] = null, options: null | object = null) {
        this.datasetInfo = { size: 0, usersNum: 0, itemsNum: 0 }
        this.datasetInfo.size = ratings.flatten().shape[0];

        let randomTen = Array.from(tf.util.createShuffledIndices(this.datasetInfo.size));
        items = items.reshape([-1, 1]).gather(randomTen);
        users = users.reshape([-1, 1]).gather(randomTen);
        ratings = ratings.flatten().gather(randomTen);
        // this.datasetInfo.usersNum = tf.unique(items)["values"].shape[0];   
        // this.datasetInfo.itemsNum = tf.unique(users)["values"].shape[0];   

        if (validationPercentage > 0) {
            this.splitTrainValidTensor(items, users, ratings, validationPercentage)
        }
        else {
            let psuedoTrainingDataset = []
            for (let i = 0; i < ratings.shape[0]; i++) {
                psuedoTrainingDataset.push({ xs: { user: users.slice(i, 1), item: items.slice(i, 1) }, ys: { rating: ratings.slice(i) } })
            }
            this.trainingDataset = tf.data.array(psuedoTrainingDataset)
        }


    }


    async getInfoOnCsv(path: string, userColumn: string, itemColumn: string) {
        let datasetSize_ = new Promise<IdatasetInfo>(function (resolve, reject) {
            let csvInfo = { size: 0, usersNum: 0, itemsNum: 0 }
            let uniqueItems = new Set()
            let uniqueUsers = new Set()
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
                }
                )
        });
        return datasetSize_;
    }

    splitTrainValidTensor(items, users, ratings, validationPercentage: number) {
        let trainSize: number = Math.round((1 - validationPercentage) * this.datasetInfo.size)
        let validSize: number = Math.abs(trainSize - this.datasetInfo.size)
        let [trainingItems, validationItems] = tf.split(items, [trainSize, validSize], 0);
        let [trainingUsers, validationUsers] = tf.split(users, [trainSize, validSize], 0);
        let [trainingRatings, validationRatings] = tf.split(ratings, [trainSize, validSize], 0);

        let psuedoTrainingDataset = []
        for (let i = 0; i < trainingRatings.shape[0]; i++) {
            psuedoTrainingDataset.push({ xs: { user: trainingUsers.slice(i, 1), item: trainingItems.slice(i, 1) }, ys: { rating: trainingRatings.slice(i) } })
        }
        this.trainingDataset = tf.data.array(psuedoTrainingDataset)


        let psuedoValidationDataset = []
        for (let i = 0; i < validationRatings.shape[0]; i++) {
            psuedoValidationDataset.push({ xs: { user: validationUsers.slice(i, 1), item: validationItems.slice(i, 1) }, ys: { rating: validationRatings.slice(i) } })
        }
        this.validationDataset = tf.data.array(psuedoValidationDataset)
    }
}
