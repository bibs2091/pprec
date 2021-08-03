import { DataBlock } from '../lib/DataBlock'
import { Learner } from '../lib/Learner'

let IMDB = new DataBlock()
IMDB.fromCsv("./examples/data.csv", 'user', 'movie', 'rating')
let learner = new Learner(IMDB, 944, 1683, 5e-3)
learner.fit(1)

