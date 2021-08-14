import * as tf from '@tensorflow/tfjs-node'
import * as csv from '@fast-csv/parse';
import * as fs from 'fs';

interface IdatasetInfo{
    size: number; usersNum: number; itemsNum: number;
}

export class DataBlock {
    trainingDataset: tf.data.Dataset<any>;
    validationDataset: tf.data.Dataset<any>;
    datasetInfo: IdatasetInfo;
    ratingRange: null | number[];
    async fromCsv(path: string, userColumn: string, itemColumn: string, ratingColumn: string, validationPercentage: number = 0.2, delimiter: string = ',', batchSize: number = 16, ratingRange: null | number[] = null, seed: number = 42, options: null | object = null) {
        let myPath = "file://" + path;
        this.datasetInfo = await this.getInfoOnCsv(path)
        // console.log(this.datasetInfo );
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
        items = items.reshape([-1, 1])
        users = users.reshape([-1, 1])
        ratings = ratings.flatten()
        let psuedoDataset: any[] = []
        for (let i = 0; i < ratings.shape[0]; i++) {
            psuedoDataset.push({ xs: { user: users.slice(i, 1), item: items.slice(i, 1) }, ys: { rating: ratings.slice(i) } })
        }
        this.trainingDataset = tf.data.array(psuedoDataset)
    }


    async getInfoOnCsv(path: string) {
        let datasetSize_ = new Promise<IdatasetInfo>(function (resolve, reject) {
            let csvInfo = { size: 0, usersNum: 0, itemsNum: 0 }
            let uniqueItems = new Set()
            let uniqueUsers = new Set()
            csv.parseFile(path, { headers: true })
                .on('error', error => console.error(error))
                .on('data', (data) => {
                    uniqueUsers.add(data.user)
                    uniqueItems.add(data.movie)
                })
                .on('end', (rowCount: number) => {
                    csvInfo.size = rowCount;
                    csvInfo.usersNum = uniqueUsers.size + 1
                    csvInfo.itemsNum = uniqueItems.size + 1
                    return resolve(csvInfo);
                }
                )
        });
        return datasetSize_;
    }

}
