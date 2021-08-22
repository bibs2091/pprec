# pprec
A privacy preserving recommender system node-js package

To run the project you only need to install the package in your project:
```
npm install pprec
```
# Example
```
import { DataBlock, Learner } from 'pprec';

async function main() {
    let IMDB = new DataBlock();

    await IMDB.fromCsv("./examples/data.csv", 'user', 'movie', 'rating');

    let learner = new Learner(IMDB, 5e-2);

    //train the model for 3 epoches
    await learner.fit(3);

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
- [ ] Local DP
- [ ] Documentation
- [ ] Contribution guide
- [ ] Output DP
- [ ] Gradient perturbation
- [ ] Other algorthims than Matrix factorization



