export function generelizedRandomResponse(x: number, ratingRange: number[], epsilon: number = 0.9): number {
    let p : number = Math.exp(epsilon) / (Math.exp(epsilon) + (ratingRange[ratingRange.length - 1] - ratingRange[0]));
    let q :number = 1 / (Math.exp(epsilon) + (ratingRange[ratingRange.length - 1] - ratingRange[0]));
    let probablities: number[] = []
    for (let i = ratingRange[0]; i <= ratingRange[ratingRange.length - 1]; i++) {
        if (i == x) probablities.push(p)
        else probablities.push(q)
    }
    let result: number = weightedChoice(probablities);
    return result;

}


function weightedChoice(p: number[]): number {
    let rnd = p.reduce((a, b) => a + b) * Math.random();
    return p.findIndex(a => (rnd -= a) < 0);
}


