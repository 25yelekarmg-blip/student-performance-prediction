const form = document.getElementById("predictionForm");
const clearButton = document.getElementById("clearButton");
const resultBox = document.getElementById("resultBox");
const predictionText = document.getElementById("predictionText");
const scoreText = document.getElementById("scoreText");

function gradeToScore(grade) {
    if (grade === "A") {
        return 90;
    }

    if (grade === "B") {
        return 75;
    }

    if (grade === "C") {
        return 60;
    }

    return 45;
}

function calculatePerformanceScore(student) {
    let participationBonus = 0;

    if (student.participation === "Yes") {
        participationBonus = 6;
    }

    return (
        student.attendance * 0.20
        + student.assignment * 0.20
        + student.midterm * 0.25
        + gradeToScore(student.previousGrade) * 0.20
        + student.studyHours * 4
        + participationBonus
    );
}

function getPrediction(score) {
    if (score >= 78) {
        return "Good";
    }

    if (score >= 58) {
        return "Average";
    }

    return "Poor";
}

function showPrediction(prediction, score) {
    predictionText.className = "";

    if (prediction === "Good") {
        predictionText.classList.add("good");
    } else if (prediction === "Average") {
        predictionText.classList.add("average");
    } else {
        predictionText.classList.add("poor");
    }

    predictionText.textContent = prediction;
    scoreText.textContent = "Performance score: " + score.toFixed(2);
    resultBox.classList.remove("hidden");
}

form.addEventListener("submit", function (event) {
    event.preventDefault();

    const student = {
        attendance: Number(document.getElementById("attendance").value),
        assignment: Number(document.getElementById("assignment").value),
        midterm: Number(document.getElementById("midterm").value),
        previousGrade: document.getElementById("previousGrade").value,
        studyHours: Number(document.getElementById("studyHours").value),
        participation: document.getElementById("participation").value
    };

    const score = calculatePerformanceScore(student);
    const prediction = getPrediction(score);

    showPrediction(prediction, score);
});

clearButton.addEventListener("click", function () {
    form.reset();
    resultBox.classList.add("hidden");
});
