import { DataBlock } from '../lib/DataBlock'
import { Learner } from '../lib/Learner'

async function main() {
    let IMDB = new DataBlock()
    await IMDB.fromCsv("./examples/data.csv", 'user', 'movie', 'rating')
    let learner = new Learner(IMDB, 5e-3)

    await learner.fit(1);
    // learner.recommendItem(10).print();

    // // saving the model
    // learner.save("myModel");

}

main()

