const dfd = require("danfojs-node")
import * as tf from '@tensorflow/tfjs-node'
import * as csv from '@fast-csv/parse';
import * as fs from 'fs';


export class DataBlock {
    trainingDataset: tf.data.Dataset<any>;
    validationDataset: tf.data.Dataset<any>;
    datasetSize: number;
    ratingRange: null | number[];
    async fromCsv(path: string, userColumn: string, itemColumn: string, ratingColumn: string, validationPercentage: number = 0.2, header: boolean = true, delimiter: string = ',', batchSize: number = 16, ratingRange: null | number[] = null, seed: number = 42, options: null | object = null) {
        let myPath = "file://" + path;
        this.datasetSize = await this.getInfoOnCsv(path, header)
        this.ratingRange = ratingRange;


        let csvDataset: tf.data.Dataset<any> = (tf.data.csv(
            myPath, {
            configuredColumnsOnly: true,
            delimiter: delimiter,
            hasHeader: header,
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
        })).shuffle(this.datasetSize, seed.toString(), false)


        let trainSize = Math.round((1 - validationPercentage) * this.datasetSize)

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


    async getInfoOnCsv(path: string, header: boolean) {
        let datasetSize_ = new Promise<number>(function (resolve, reject) {
            csv.parseFile(path)
                .on('error', error => console.error(error))
                .on('data', () => { })
                .on('end', (rowCount: number) => resolve(rowCount))
        });
        return (await datasetSize_) - (header ? 1 : 0);
    }

}
