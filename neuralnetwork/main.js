function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

function sigmoidDerivative(x) {
    return x * (1 - x);
}

function shuffle(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
}

class Perceptron {
    constructor(inputSize, outputSize, hiddenOutputSize1, hiddenOutputSize2, learningWeight = 0.1) {
        this.inputSize = inputSize;
        this.hiddenOutputSize1 = hiddenOutputSize1;
        this.hiddenOutputSize2 = hiddenOutputSize2;
        this.outputSize = outputSize;
        this.learningWeight = learningWeight;

        this.weights_i_h1 = Array.from({ length: hiddenOutputSize1 }, () =>
            Array.from({ length: inputSize }, () => Math.random() * 2 - 1)
        );

        this.weights_h1_h2 = Array.from({ length: hiddenOutputSize2 }, () =>
            Array.from({ length: hiddenOutputSize1 }, () => Math.random() * 2 - 1)
        );

        this.weights_o = Array.from({ length: outputSize }, () =>
            Array.from({ length: hiddenOutputSize2 }, () => Math.random() * 2 - 1)
        );

        this.biases_i_h1 = Array.from({ length: hiddenOutputSize1 }, () => Math.random() * 2 - 1);
        this.biases_h1_h2 = Array.from({ length: hiddenOutputSize2 }, () => Math.random() * 2 - 1);
        this.biases_o = Array.from({ length: outputSize }, () => Math.random() * 2 - 1);
    }

    predict(input) {
        let outputs_i_h1 = new Array(this.hiddenOutputSize1);

        for (let i = 0; i < this.hiddenOutputSize1; ++i) {
            let sum = this.biases_i_h1[i];
            for (let x = 0; x < this.inputSize; ++x) {
                sum += this.weights_i_h1[i][x] * input[x];
            }

            outputs_i_h1[i] = sigmoid(sum);
        }

        let outputs_h1_h2 = new Array(this.hiddenOutputSize2);

        for (let i = 0; i < this.hiddenOutputSize2; ++i) {
            let sum = this.biases_h1_h2[i];
            for (let x = 0; x < this.hiddenOutputSize1; ++x) {
                sum += this.weights_h1_h2[i][x] * outputs_i_h1[x];
            }

            outputs_h1_h2[i] = sigmoid(sum);
        }

        let outputs = new Array(this.outputSize);

        for (let i = 0; i < this.outputSize; ++i) {
            let sum = this.biases_o[i];
            for (let x = 0; x < this.hiddenOutputSize2; ++x) {
                sum += this.weights_o[i][x] * outputs_h1_h2[x];
            }

            outputs[i] = sigmoid(sum);
        }

        return outputs;
    }

    train(input, target) {
        let outputs_i_h1 = new Array(this.hiddenOutputSize1);

        for (let i = 0; i < this.hiddenOutputSize1; ++i) {
            let sum = this.biases_i_h1[i];
            for (let x = 0; x < this.inputSize; ++x) {
                sum += this.weights_i_h1[i][x] * input[x];
            }

            outputs_i_h1[i] = sigmoid(sum);
        }

        let outputs_h1_h2 = new Array(this.hiddenOutputSize2);

        for (let i = 0; i < this.hiddenOutputSize2; ++i) {
            let sum = this.biases_h1_h2[i];
            for (let x = 0; x < this.hiddenOutputSize1; ++x) {
                sum += this.weights_h1_h2[i][x] * outputs_i_h1[x];
            }

            outputs_h1_h2[i] = sigmoid(sum);
        }

        let outputs = new Array(this.outputSize);

        for (let i = 0; i < this.outputSize; ++i) {
            let sum = this.biases_o[i];
            for (let x = 0; x < this.hiddenOutputSize2; ++x) {
                sum += this.weights_o[i][x] * outputs_h1_h2[x];
            }

            outputs[i] = sigmoid(sum);
        }

        let outputErrors = new Array(this.outputSize);
        let outputDelta = new Array(this.outputSize);
        for (let i = 0; i < this.outputSize; ++i) {
            outputErrors[i] = target[i] - outputs[i];
            outputDelta[i] = outputErrors[i] * sigmoidDerivative(outputs[i]);
        }

        let hiddenErrors2 = new Array(this.hiddenOutputSize2).fill(0);
        let hiddenDelta2 = new Array(this.hiddenOutputSize2);
        for (let h = 0; h < this.hiddenOutputSize2; h++) {
            for (let o = 0; o < this.outputSize; o++) {
                hiddenErrors2[h] += outputDelta[o] * this.weights_o[o][h];
            }
            hiddenDelta2[h] = hiddenErrors2[h] * sigmoidDerivative(outputs_h1_h2[h]);
        }

        let hiddenErrors1 = new Array(this.hiddenOutputSize1).fill(0);
        let hiddenDelta1 = new Array(this.hiddenOutputSize1);
        for (let h = 0; h < this.hiddenOutputSize1; h++) {
            for (let o = 0; o < this.hiddenOutputSize2; o++) {
                hiddenErrors1[h] += hiddenDelta2[o] * this.weights_h1_h2[o][h];
            }
            hiddenDelta1[h] = hiddenErrors1[h] * sigmoidDerivative(outputs_i_h1[h]);
        }

        for (let o = 0; o < this.outputSize; o++) {
            for (let h = 0; h < this.hiddenOutputSize2; h++) {
                this.weights_o[o][h] += this.learningWeight * outputDelta[o] * outputs_h1_h2[h];
            }
            this.biases_o[o] += this.learningWeight * outputDelta[o];
        }

        for (let h2 = 0; h2 < this.hiddenOutputSize2; h2++) {
            for (let h1 = 0; h1 < this.hiddenOutputSize1; h1++) {
                this.weights_h1_h2[h2][h1] += this.learningWeight * hiddenDelta2[h2] * outputs_i_h1[h1];
            }
            this.biases_h1_h2[h2] += this.learningWeight * hiddenDelta2[h2];
        }

        for (let h1 = 0; h1 < this.hiddenOutputSize1; h1++) {
            for (let i = 0; i < this.inputSize; i++) {
                this.weights_i_h1[h1][i] += this.learningWeight * hiddenDelta1[h1] * input[i];
            }
            this.biases_i_h1[h1] += this.learningWeight * hiddenDelta1[h1];
        }
    }
}

async function updateArray(lastX, lastY, currentX, currentY) {
    const dx = currentX - lastX;
    const dy = currentY - lastY;
    const steps = Math.max(Math.abs(dx), Math.abs(dy));

    for (let i = 0; i <= steps; i++) {
        const x = Math.round(lastX + (dx * i) / steps);
        const y = Math.round(lastY + (dy * i) / steps);

        if (x >= 0 && x < WEIGHT && y >= 0 && y < HEIGHT) {
            array[y * WEIGHT + x] = 1;
        }
    }
}


document.addEventListener("mousemove", (event) => {
    if (drawing) {
        const currentX = event.offsetX;
        const currentY = event.offsetY;

        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(currentX, currentY);
        ctx.stroke();

        updateArray(lastX, lastY, currentX, currentY);

        lastX = currentX;
        lastY = currentY;
    } else {
        return;
    }
});

document.addEventListener("mouseup", (event) => {
    drawing = false;
});

document.addEventListener("mousedown", (event) => {
    drawing = true;

    lastX = event.offsetX;
    lastY = event.offsetY;
});

document.addEventListener("keypress", (event) => {
    if (event.key.toLowerCase() === "p") {
        goodStatus = !goodStatus;
        badStatus = !goodStatus;
    } else if (event.key.toLowerCase() === "v") {
        if (goodStatus === false) {
            alert("Add new example in good");
            good.push({ input: [...array], target: [1, 0] });
            array.fill(0);
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.beginPath();
        } else {
            alert("Add new example in bad");
            bad.push({ input: [...array], target: [0, 1] });
            array.fill(0);
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.beginPath();
        }
    } else if (event.key.toLowerCase() === "r") {
        if (good.length !== 0 && bad.length !== 0) {
            const result = net.predict(array);

            if (result[0] > result[1]) {
                alert("I think what you draw GOOD");
            } else {
                alert("I think what you draw BAD");
            }

            array.fill(0);
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.beginPath();
        }
    } else if (event.key.toLowerCase() === "t") {
        console.log(good.length);
        console.log(bad.length);
        if (good.length === 0 || bad.length === 0) {
            alert("Добавьте примеры и для GOOD и для BAD");
            return;
        }
        const trainData = [...good, ...bad];
        const initialLearningWeight = net.learningWeight;

        for (let epoch = 0; epoch < 5000; epoch++) {
            if (epoch % 100 === 0) {
                net.learningWeight = initialLearningWeight * Math.pow(0.95, epoch / 100);
            }
            shuffle(trainData);
            for (const sample of trainData) {
                net.train(sample.input, sample.target);
            }
        }
        alert("Обучение завершено");
    }
});

const canvas = document.getElementById("MyCanvas");
const ctx = canvas.getContext("2d");

const WEIGHT = 28;
const HEIGHT = 28;

drawing = false;
goodStatus = false;
badStatus = false;

let lastX = 0;
let lastY = 0;

let array = new Array(WEIGHT * HEIGHT).fill(0);

let good = [];
let bad = [];

net = new Perceptron(array.length, 2, 128, 64);