# pprec
A recommender system package, easy to use in your node-js application. PPREC can also preserve the privacy of the users data.

To run the project you only need to install the package in your project:
```
npm install pprec
```
# Usage

```
import { DataBlock, Learner } from 'pprec';

async function main() {
    let IMDB = new DataBlock();

    await IMDB.fromCsv("./examples/data.csv", 'user', 'movie', 'rating');

    let learner = new Learner(IMDB, 5e-2);

    //train the model for 3 epoches
    await learner.fit(3);

    //add a new rating in the dataset user id = 5, item id = 10 and with a rating = 2.
    learner.addRating(5, 10, 2) 


    // recommend an item for the user with ID = 10
    learner.recommendItem(10).print(); 

    //get the similar 10 users to user with ID = 10
    console.log(learner.mostSimilarUsers(10)); 

    //get the similar 10 users to user with ID = 313 (titanic)
    console.log(learner.mostSimilarItems(313));

    
    learner.save("myModel"); // saving the model
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



