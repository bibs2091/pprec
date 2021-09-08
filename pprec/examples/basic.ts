import { DataBlock, Learner } from '../src/main'


async function main() {
    const IMDB = await new DataBlock().fromCsv("./examples/data.csv", 'user', 'movie', 'rating', 64, [0.8, 5.2]); //parsing the dataset from csv
    

    const learner = new Learner(IMDB, 1e-3); //Creating Learner from the IMDB datablock    
    
    await learner.fit(1); //train the mode for one epoch

    learner.recommendItem(10).print(); // recommend an item for the user with ID = 10
    learner.addRating(5, 10, 2) //add a new rating in the dataset


    console.log(learner.mostSimilarUsers(10)); //get the similar 10 users to user with ID = 10
    console.log(learner.mostSimilarItems(313)); //get the similar 10 users to user with ID = 313 (titanic)
    learner.save("myModel"); // saving the model
}

main()

