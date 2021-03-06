import { DataBlock, Learner } from '../src/main'


async function main() {
    let IMDB = new DataBlock()
    await IMDB.fromCsv("./examples/data.csv", 'user', 'movie', 'rating') //parsing the dataset from csv

    let learner = new Learner(IMDB, 5e-3); //Creating Learner from the IMDB datablock
    await learner.fit(4); //train the mode for one epoch

    learner.recommendItem(10).print(); // recommend an item for the user with ID = 10
    console.log(learner.mostSimilarUsers(10)); //get the similar 10 users to user with ID = 10
    console.log(learner.mostSimilarItems(313)); //get the similar 10 users to user with ID = 313 (titanic)
    learner.save("myModel"); // saving the model
}

main()

