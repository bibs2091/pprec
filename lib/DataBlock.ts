import * as tf from '@tensorflow/tfjs-node'

export class DataBlock {
    validationPercentage!: number;
    dataSet!: tf.data.Dataset<any>;
    fromCsv(path: string, userColumn: string, itemColumn: string, ratingColumn: string, validationPercentage: number = 0, header: boolean = true, delimiter: string = ',', batchSize: number = 32, ratingRange: null | number[] = null, randomSeed: null | number = null, options: null | object = null) {
        path = "file://" + path;
        this.validationPercentage = validationPercentage
        let csvDataset: tf.data.Dataset<any> = (tf.data.csv(
            path, {
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
        })).batch(batchSize)
        this.dataSet = csvDataset.map(x => ({ xs: { user: x.xs[userColumn].reshape([-1, 1]), item: x.xs[itemColumn].reshape([-1, 1]) }, ys: x.ys }))
    }

    fromTensor(items: tf.Tensor, users: tf.Tensor, ratings: tf.Tensor, validationPercentage: number = 0, batchSize: number = 32, ratingRange: null | number[] = null, randomSeed: null | number[] = null, options: null | object = null) {
        items = items.reshape([-1, 1])
        users = users.reshape([-1, 1])
        ratings = ratings.flatten()
        let psuedoDataset: any[] = []
        for (let i = 0; i < ratings.shape[0]; i++) {
            psuedoDataset.push({ xs: { user: users.slice(i, 1), item: items.slice(i, 1) }, ys: { rating: ratings.slice(i) } })
        }
        this.dataSet = tf.data.array(psuedoDataset)
    }



}

// dataset.fromCsv("data.csv", 'user', 'movie', 'rating')
// dataset.fromTensor(tf.tensor([1, 2, 3, 4]), tf.tensor([1, 2, 3, 4]), tf.tensor([1, 2, 3, 4]))
