import { dataBlock, learner } from '../src/main'


async function main() {
    const IMDB = await dataBlock().fromCsv("./examples/data.csv", {
        userColumn: 'user', itemColumn: 'movie', ratingColumn: 'rating', batchSize: 64, ratingRange: [1, 5]
    }); //parsing the dataset from csv

    // const IMDB = await dataBlock().fromCsv("./examples/books-ratings.csv", {
    //     userColumn: 'User-ID', itemColumn: 'ISBN', ratingColumn: 'Book-Rating', batchSize: 64,
    //     ratingRange: [0, 10],
    // });

    const myLearner = learner(IMDB, { learningRate: 1e-3 }); //Creating learner from the IMDB datablock    

    await myLearner.fit(1); //train the mode for one epoch

    console.log(myLearner.recommendItems(10, 7)); // recommend 7 items for the user with ID = 10 
    myLearner.addRating(5, 10, 2) //add a new rating in the dataset
    myLearner.newUser("22222") //add a new user with the id 22222
    console.log(myLearner.mostSimilarUsers(10)); //get the similar 10 users to user with ID = 10
    console.log(myLearner.mostSimilarItems(313)); //get the similar 10 users to user with ID = 313 (titanic)
    await myLearner.save("myModel"); // saving the model
}

main()

