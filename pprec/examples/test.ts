import { dataBlock, learner } from '../src/main'


async function main() {
   
    const myLearner = await learner({ learningRate: 1e-3 }); //Creating learner from the IMDB datablock    
     myLearner.newUser(1);
     myLearner.newUser(2);
     myLearner.newItem(1134);
     myLearner.newItem(1132);
     myLearner.newItem(1131);
     myLearner.newItem(1139);
     myLearner.newItem(1135);
     myLearner.newItem(11311);
     console.log(await myLearner.recommendItems(1,3))
    await myLearner.addRating(1,1132,5)
    console.log(await myLearner.recommendItems(1,3))
    // console.log(await myLearner.mostSimilarUsers(1,5))
}

main()

