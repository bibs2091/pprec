import * as tf from '@tensorflow/tfjs-node'
const numUsers = 944; const numMovies = 1683;
function createModel(numUsers = 944, numMovies = 1683) {
  const userInputLayer = tf.input({ shape: 1, dtype: "int32", name: "user" });
  const userEmbeddingLayer = tf.layers.embedding({
    inputDim: numUsers + 1,
    outputDim: 5,
    inputLength: 1,
    name: "userEmbeddingLayer",
  }).apply(userInputLayer)
  const userEmbeddingLayerOutput = tf.layers.flatten({ name: "flat1" }).apply(userEmbeddingLayer);

  const itemInputLayer = tf.input({ shape: [1], dtype: "int32", name: "movie" });
  const itemEmbeddingLayer = tf.layers.embedding({
    inputDim: numMovies + 1,
    outputDim: 5,
    inputLength: 1,
    name: "itemEmbeddingLayer",
  }).apply(itemInputLayer);
  const itemEmbeddingLayerOutput = tf.layers.flatten({ name: "flat2" }).apply(itemEmbeddingLayer);
  const dotLayer = tf.layers.dot({ axes: -1, name: "rating" }).apply([userEmbeddingLayerOutput, itemEmbeddingLayerOutput]);
  const model = tf.model({ inputs: [userInputLayer, itemInputLayer], outputs: dotLayer });
  const optimizer = tf.train.adam(5e-3);
  model.compile({ 
    optimizer: optimizer,
    loss: 'meanSquaredError'
  });
  // model.summary()
  return model;
}
function createDataset(csvFile = "./examples/data.csv") {
  csvFile = "file://" + csvFile;
  const csvDataset = (tf.data.csv(
    csvFile, {
    configuredColumnsOnly: true,
    columnConfigs: {
      user: {
        required: true,
        dtype: "float32"
      },
      movie: {
        required: true,
        dtype: "float32"
      },
      rating: {
        isLabel: true,
        dtype: "float32"
      }
    }
  })).batch(1024)
  const dataSet = csvDataset.map(x => ({ xs: { user: x.xs.user.reshape([-1, 1]), movie: x.xs.movie.reshape([-1, 1]) }, ys: x.ys }))
  return dataSet;
}

async function trainModel(model, dataSet, epochsNum = 1) {
  console.log("Start training");
  await model.fitDataset(dataSet, {
    epochs: epochsNum,
  });
}

function recommendItem(userId, model) {
  let toPredict = [tf.fill([numMovies, 1], userId), tf.range(0, numMovies).reshape([-1, 1])]
  return model.predictOnBatch(toPredict).argMax();
}

async function addRating(userId, movieId, rating, dataSet) {
  let toAdd = tf.data.array([{ xs: { user: tf.tensor2d([[userId]]), movie: tf.tensor2d([[movieId]]) }, ys: { rating: tf.tensor1d([rating]) } }, ])
  await dataSet.concatenate(dataSet).forEachAsync(e => console.log(e));
  return dataSet.concatenate(dataSet);
}

async function nn() {
  const model = createModel();
  let dataSet = createDataset();
  // await addRating(15,15,22,dataSet)
  await trainModel(model, dataSet,3);
  await recommendItem(9, model,3).print();
}

nn()