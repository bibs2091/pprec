import { DataBlock } from './DataBlock'
import { Learner } from './Learner'

export function dataBlock() {
    return new DataBlock();
}

export function learner(dataBlockargs, options) {
    return new Learner(dataBlockargs, options);
}