import { learner } from '../src/main'


async function main() {

    const myLearner = learner(); //Creating learner from the IMDB datablock    
    await myLearner.load("myModel"); //load an already saved model
    myLearner.recommendItem(10).print(); // recommend an item for the user with ID = 10
}

main()

