import * as tf from '@tensorflow/tfjs-node'

class DataBlock {

    fromCsv(path, itemColumn, userColumn, ratingColumn, validationPercentage = 0, header = true, delimiter = ',', batchSize = 32, ratingRange = null, randomSeed = null, options = null) {
        path = "file://" + path;
        this.validationPercentage = validationPercentage
        let csvDataset = (tf.data.csv(
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

    fromTensor(items, users, ratings, validationPercentage = 0, batchSize = 32, ratingRange = null, randomSeed = null, options = null) {
        items = items.reshape([-1, 1])
        users = users.reshape([-1, 1])
        ratings = ratings.flatten()
        let psuedoDataset = []
        for (let i = 0; i < ratings.shape[0]; i++) {
            psuedoDataset.push({ xs: { user: users.slice(i, 1), item: items.slice(i, 1) }, ys: { rating: ratings.slice(i) } })
        }
        this.dataSet = tf.data.array(psuedoDataset)
    }


 
}

let dataset = new DataBlock()
// dataset.fromCsv("data.csv", 'user', 'movie', 'rating')
// dataset.fromTensor(tf.tensor([1, 2, 3, 4]), tf.tensor([1, 2, 3, 4]), tf.tensor([1, 2, 3, 4]))
