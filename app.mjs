import * as tf from '@tensorflow/tfjs-node'

function createModel(numMovies = 1683, numUsers = 944) {
  const userInputLayer = tf.input({ shape: 1, dtype: "int32", name: "user" });
  const userEmbeddingLayer = tf.layers.embedding({
    inputDim: numUsers + 1,
    outputDim: 5,
    inputLength: 1,
    name: "userEmbeddingLayer",
  }).apply(userInputLayer)
  const userEmbeddingLayerOutput = tf.layers.flatten({ name: "flat1" }).apply(userEmbeddingLayer);

  const itemInputLayer = tf.input({ shape: 1, dtype: "int32", name: "movie" });
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
function createDataset(csvFile = "data.csv") {
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
  })).batch(1)
  return csvDataset;
}

async function trainModel(model, dataSet) {
  console.log("Start training");
  let val = await model.fitDataset(dataSet, {
    epochs: 10,
  });
}

async function mmm() {
  const model = await createModel();
  const dataSet = await createDataset();

  // dataSet.filter(x => x.ys.rating === 3 ).toArray().then(a => console.log(a));
  // await dataSet.toArray().then(a => console.log(a.xs.user.dataSync()));
  const dataSet2 = await dataSet.map(x => ({ xs: { user: x.xs.user.reshape([1, 1]), movie: x.xs.movie.reshape([1, 1]) }, ys: x.ys }))
  // await dataSet2.forEachAsync(e => console.log(e.xs.user));

  // dataSet2.toArray().then(a => console.log(a));
  await trainModel(model, dataSet2);
}

mmm()


// async function m() {
//   const xArray = [
//     [1, 1, 1, 1, 1, 1, 1, 1, 1],
//     [1, 1, 1, 1, 1, 1, 1, 1, 1],
//     [1, 1, 1, 1, 1, 1, 1, 1, 1],
//     [1, 1, 1, 1, 1, 1, 1, 1, 1],
//   ];
//   const yArray = [1, 1, 1, 1];
//   const xDataset = tf.data.array(xArray);
//   const yDataset = tf.data.array(yArray);
//    const xyDataset = tf.data.zip({ xs: xDataset, ys: yDataset })
//      .batch(4)
//      xyDataset.toArray().then(a => console.log(a));

//    const model = tf.sequential({
//      layers: [tf.layers.dense({ units: 1, inputShape: [9] })]
//    });
//    model.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });
//    const history = await model.fitDataset(xyDataset, {
//      epochs: 4,
//      callbacks: { onEpochEnd: (epoch, logs) => console.log(logs.loss) }
//    });

//  }
 //  m()