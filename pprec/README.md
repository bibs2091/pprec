# pprec
PPREC is node-js library made for web applications to recommend items to users. The library offer a higher level of abstraction for developers who are not comfortable with concepts like tensors, layers, optimizersand loss functions, and want to add a recommender in few lines of code.

To run the project you only need to install the package in your project:
```
npm install pprec
```
# Usage

```
import { dataBlock, learner } from 'pprec';

async function main() {
    const IMDB = await dataBlock().fromCsv("./examples/data.csv", {
        userColumn: 'user', itemColumn: 'movie', ratingColumn: 'rating',batchSize: 64, ratingRange: [1, 5]});

    const myLearner = learner(IMDB, { learningRate: 1e-3 });

    await myLearner.fit(3);

    //add a new rating in the dataset user id = 5, item id = 10 and with a rating = 2.
    myLearner.addRating(5, 10, 2) 


    // recommend 7 items for the user with ID = 10 
    console.log(myLearner.recommendItems(10, 7)); 

    //get the similar 10 users to user with ID = 10
    console.log(myLearner.mostSimilarUsers(10)); 

    //get the similar 10 users to user with ID = 313 (titanic)
    console.log(myLearner.mostSimilarItems(313));
    
    myLearner.save("myModel"); // saving the model
}

main()
```

The current progress:
- [x] Learner 
- [x] DataBlock
- [x] Local DP
- [ ] Documentation
- [ ] Contribution guide
- [ ] Output DP
- [ ] Gradient perturbation
- [ ] Other algorthims than Matrix factorization



