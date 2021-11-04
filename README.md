# pprec
PPREC is node-js library made for web applications to help them integrate a recommendation systems easily. The library offer a higher level of abstraction for developers who are not comfortable with concepts like tensors, layers, optimizers and loss functions, and want to add a recommender in few lines of code.

To run the project you need to have redis installed then start it:
```
redis-server
```
and install the package in your project:
```
npm install pprec
```
# Usage

## Load data
if you don't have the data yet to use for training jump to [Without DataBlock](#Without-DataBlock)
 . 

You can either load data in pprec from a csv file or existing tensors:
* CSV file: Specify the columns names that contains the information about the users, items, and ratings. 
```
const data = await dataBlock().fromCsv("data.csv", {
        userColumn: 'user',
        itemColumn: 'movie', 
        ratingColumn: 'rating',
        batchSize: 64, 
        ratingRange: [0, 5]
        });
```
* Tensor: 
```
const data = dataBlock().fromTensor(
    items = tf.tensor([10,7,3,10],
    users = tf.tensor([15,30,1,500],
    ratings = tf.tensor([1,2,2,5]),
    batchSize =  4,
    ratingRange= [0, 5]);
```

## Creating a Learner
Learner is the responsible for training the recommendation model and infrencing/generating recommendations from it.
To create a learner:
```
const myLearner = learner(data, { learningRate: 1e-3 });
``` 
then fit it for few epoches:
```
await myLearner.fit(3);
``` 
## Adding a rating 
pprec supports online learning so it will allow adding a new rating to the dataset and adjust the learner to it:
```
await myLearner.addRating("MohamedNaas001", "The office", 5);
```
You do not need to run  myLearner.fit() again, as it is already embedded in the addRating() method.
## Adding new user/item
In case there is a new user or item in your system, you should explicitly inform pprec before trying to add recommendations and generating recommendations to them:
```
myLearner.newUser("UUID25435") //add a new user with the id UUID25435

myLearner.newItem("Squid Games") //add a new user with the id UUID25435
```
The new user/item latent factors (embeddings) will be the average of the existing latent factors for all the existing users/items. 

##  Generating recommendation
To generate **k** items recommendations for a user just do this
```
console.log(myLearner.recommendItems("MohamedNaas001", 7)); 
//recommend 7 items for the user with ID = "MohamedNaas001" 
```
By default, the recommendation will not be repeated, this means if a user already viewed or rated an item it will be saved in *redis* to be eliminated in the recommendation process.

To tell pprec that a user viewed an item:
```
myLearner.viewed("MohamedNaas001", "Dark")
```
viewing an item means that the user viewed the item but it did not rate it.

## Similar items/users
You can get the **k** similar items to an item or users to a user using the cosine similarity between the items/users latent factors:
```
console.log(myLearner.mostSimilarUsers(""MohamedNaas001""));

console.log(myLearner.mostSimilarItems("House MD"));
```

## Saving and Loading Learner
To save a the trained learner
```
myLearner.save("myModel"); 
```
To load a learner
```
await myLearner.load("myModel"); 
```
## Saving an existing DataBlock
To save a datablock in csv format:
```
await data.save("IMDB.csv")
```
You can use the DataBlock.fromCsv() method to load the data in pprec again.

## Without DataBlock
pprec takes into account the case when a website does not have any data to build the recommendation on, in this case you can initilize the Learner directly then add users, items, and ratings to it. Example
```
const myLearner = learner({ learningRate: 1e-3 });

myLearner.newUser("UUID25435");

myLearner.newItem("Squid Games");

await myLearner.addRating("UUID25435", "Squid Games", 4);
```



# The current progress:
- [x] Learner 
- [x] DataBlock
- [x] Local DP
- [x] Documentation: https://pprec.netlify.app/
- [ ] Contribution guide
- [ ] Output DP
- [ ] Gradient perturbation
- [ ] Other algorthims than Matrix factorization



