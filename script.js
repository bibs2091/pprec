function countUnique(iterable) {
    return new Set(iterable).size;
  }

var train_xs = []
var train_xs2 = []
var train_ys = []
input.map(element => {
    train_xs.push([element.user])
    train_xs2.push([element.movie])
})

output.map(element => {
    train_ys.push([element.rating])
})

const numUsers = 944
const numMovies = 1683

train_xs = tf.tensor2d(train_xs);
train_xs2 = tf.tensor2d(train_xs2);
train_ys = tf.tensor2d(train_ys);

userInputLayer = tf.input({ shape: [1], dtype: "int32" });
itemInputLayer = tf.input({ shape: [1], dtype: "int32" });

var userEmbeddingLayer = tf.layers.embedding({
    inputDim: numUsers  + 1,
    outputDim: 3,
    inputLength: 1,
}).apply(userInputLayer)

userEmbeddingLayer = tf.layers.flatten().apply(userEmbeddingLayer)

var itemEmbeddingLayer = tf.layers.embedding({
    inputDim: numMovies + 1,
    outputDim: 3,
    inputLength: 1,
}).apply(itemInputLayer);

itemEmbeddingLayer = tf.layers.flatten().apply(itemEmbeddingLayer)
var dotLayer = tf.layers.dot({ axes: -1 }).apply([userEmbeddingLayer, itemEmbeddingLayer]);

var model = tf.model({ inputs: [userInputLayer, itemInputLayer], outputs: dotLayer });
const optimizer = tf.train.adam(5e-3);
model.compile({
    optimizer: optimizer,
    loss: 'meanSquaredError'
});

async function nn() {
    console.log('Start')
    for (let index = 0; index < 1; index++) {
        let val = await model.fit([train_xs, train_xs2], train_ys, {
            shuffle: true,
            batchSize: 1,
            epochs: 1
        });
        console.log(val.history.loss[0]);
    }
    console.log('end')
    // console.log(result.history.loss[0]);
}
// nn()