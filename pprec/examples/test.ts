import { dataBlock, learner } from '../src/main'


async function main() {
    const IMDB = await dataBlock().fromCsv("./examples/data.csv", {
        userColumn: 'user', itemColumn: 'movie', ratingColumn: 'rating', batchSize: 64, ratingRange: [1, 5]
    }); 
    await IMDB.test();
    const myLearner = learner(IMDB, { learningRate: 1e-3 }); //Creating learner from the IMDB datablock    
    myLearner.recommendItems(1,5)

}

main()

