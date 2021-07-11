const numUsers = 944;
const numMovies = 1683;

const model = tf.sequential();

const P = tf.sequential();
P.add(tf.layers.embedding({
    inputDim: numUsers  + 1,
    outputDim: 3,
    inputLength: 1,
}));
P.add(tf.layers.flatten());


const Q = tf.sequential();
Q.add(itemEmbeddingLayer = tf.layers.embedding({
    inputDim: numMovies + 1,
    outputDim: 3,
    inputLength: 1,
}));
Q.add(tf.layers.flatten());

model.add(P)
model.add(Q)
model.add(tf.layers.dot({ axes: -1 }))

consol.log(model)