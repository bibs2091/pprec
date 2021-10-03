import { DataBlock } from './DataBlock'
import { Learner } from './Learner'

export function dataBlock() {
    return new DataBlock();
}


export function learner(dataBlockargs?, options?) {
    if (dataBlockargs != null && options != null)
        return new Learner(dataBlockargs, options);
    else return new Learner();
}