import { DataBlock } from '../lib/DataBlock'
import { Learner } from '../lib/Learner'
import * as tf from '@tensorflow/tfjs-node'


async function main() {
    let IMDB = new DataBlock()
    await IMDB.fromCsv("./examples/data.csv", 'user', 'movie', 'rating')

    // IMDB.fromTensor(tf.tensor([1, 2, 3, 4]), tf.tensor([1, 2, 3, 4]), tf.tensor([1, 2, 3, 4]))

    let learner = new Learner(IMDB, 5e-3);
    // await learner.fit(1);
    let newUserID = learner.newUser()
    console.log("the new user id is:"  + newUserID);

    let newItemID = learner.newItem()
    console.log("the new item id is:"  + newItemID);

    await learner.fit(1);
    learner.recommendItem(10).print();

    // saving the model
    learner.save("myModel");

}

main()

