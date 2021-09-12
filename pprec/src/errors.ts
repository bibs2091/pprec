export class ValueError extends Error {
    constructor(message?: string) {
        super(message);
        Object.setPrototypeOf(this, ValueError.prototype);
    }
}

export class NonExistance extends Error {
    constructor(message?: string) {
        super(message);
        Object.setPrototypeOf(this, NonExistance.prototype);
    }
}
