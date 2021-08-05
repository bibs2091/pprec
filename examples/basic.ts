import { DataBlock } from '../lib/DataBlock'
import { Learner } from '../lib/Learner'
const dfd = require("danfojs-node")

async function main() {
    let IMDB = await new DataBlock()
    await IMDB.fromCsv("./examples/data.csv", 'user', 'movie', 'rating')
    let learner = new Learner(IMDB, 944, 1683, 5e-3)
    learner.fit(3)
}

main()
