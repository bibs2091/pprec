import { dataBlock, learner } from '../src/main'
const { performance } = require('perf_hooks');


async function main() {

    const IMDB = await dataBlock().fromCsv("./examples/books-ratings.csv", {
        userColumn: 'User-ID', itemColumn: 'ISBN', ratingColumn: 'Book-Rating', batchSize: Math.pow(2, 11),
        seed: 42
    });

    let embeddings = [1,5,10,20,30,50,75,100,125,150,200,250]
    let iterations = 100
    embeddings.forEach(e => {
        let myLearner = learner(IMDB, { learningRate: 3e-3, embeddingOutputSize: e }); //Creating learner from the IMDB datablock    
        let start = performance.now();
        for (let i = 0; i < iterations; i++)
            myLearner.addRatingSync("276725", "6", 2) //add a new rating in the dataset
        let end = performance.now();
        
        console.log("for " + e + ":");
        console.log(`${(end - start) / iterations}ms`); 

    })

    
}

main()

