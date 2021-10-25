import { dataBlock, learner } from '../src/main'


async function main() {
    const IMDB = await dataBlock().fromCsv("./examples/data.csv", {
        userColumn: 'user', itemColumn: 'movie', ratingColumn: 'rating', batchSize: 64, ratingRange: [1, 5]
    }); 
    const myLearner = learner(IMDB, { learningRate: 1e-3 }); //Creating learner from the IMDB datablock    
    // await myLearner.fit()
    console.log(await myLearner.recommendItems(1,5))
    myLearner.addRating(1,1134,5)
    console.log(await myLearner.recommendItems(1,5))
    IMDB.client.quit();
}

main()

