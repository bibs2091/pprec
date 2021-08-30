import * as tf from '@tensorflow/tfjs-node'

export function cosineSimilarity(a: tf.Tensor, b: tf.Tensor) {
    let similarity: tf.Tensor = a.dot(b.reshape([-1])).div((tf.norm(a,"euclidean", 1).mul(tf.norm(b,"euclidean", 1))));
    return similarity;
}

export function euclideandistance(a: tf.Tensor, b: tf.Tensor) {
    let similarity: tf.Tensor = tf.norm(a.sub(b),"euclidean", 1)
    return similarity;
}

