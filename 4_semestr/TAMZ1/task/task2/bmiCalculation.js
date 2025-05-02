document.addEventListener("DOMContentLoaded", function () {
    document.getElementById("unitSystem").addEventListener("ionChange", updateUnits);
});



function updateUnits() {
    const unitSystem = document.getElementById("unitSystem").value;

    if (unitSystem === "metric") {
        document.getElementById("weightUnitLabel").setAttribute("placeholder", "kg");
        document.getElementById("heightUnitLabel").setAttribute("placeholder", "cm");
    } else {
        document.getElementById("weightUnitLabel").setAttribute("placeholder", "lbs");
        document.getElementById("heightUnitLabel").setAttribute("placeholder", "in");
    }
}

function countBMI() {
    const age = document.getElementById('age').value;
    let height = document.getElementById('heightUnitLabel').value;
    let weight = document.getElementById('weightUnitLabel').value;
    const gender = document.getElementById('gender').value;
    const metric = document.getElementById('unitSystem').value;

    console.log(height, weight, gender, metric, age);

    const bmi = calculateBMI(weight, height, metric);

    console.log(bmi);

    displayBMI(bmi);

    saveToSessionStorage(age, height, weight, bmi, gender, metric);
    storeDataToLocalStoradge(height, weight, metric, bmi, gender, age);
}

function calculateBMI(weight, height, metric) {
    console.log(height, weight);
    let result = 0;

    if (metric === "metric") {
        height = height / 100;
        result = weight / (height * height);
    } else {
        result = 703 * (weight / (height * height));
    }

    console.log(result);
    return Number(result.toFixed(2));
}

function displayBMI(bmi) {
    document.getElementById('bmiResult').innerHTML = `: ${bmi.toFixed(2)}`;
    document.getElementById('classification').innerHTML = `Classification: ${clasificationBMI(bmi)}`;
    if (bmi < 18.5) {
        progress = 0.2;
        setProgressBar(0.2, "warning");
    } else if (bmi >= 18.5 && bmi < 24.9) {
        setProgressBar(0.5, "success");
    } else if (bmi >= 25 && bmi < 29.9) {
        setProgressBar(0.7, "warning");
    } else if (bmi >= 30) {
        setProgressBar(1, "danger");
    }
    document.getElementsByClassName('card-container')[0].style.display = 'block';
}

function clasificationBMI(bmi) {
    if (bmi < 16) {
        return "Severe Thinness"
    } else if (bmi < 17 && bmi > 16) {
        return "Moderate Thinness"
    } else if (bmi < 18.5 && bmi > 17) {
        return "Mild Thinness"
    } else if (bmi < 25 && bmi > 18.5) {
        return "Normal"
    } else if (bmi < 30 && bmi > 25) {
        return "Overweight"
    } else if (bmi < 35 && bmi > 30) {
        return "Obese Class I"
    } else if (bmi < 40 && bmi > 35) {
        return "Obese Class II"
    } else if (bmi > 40) {
        return "Obese Class III"
    }
}

function setProgressBar(range, color) {
    document.getElementById('bmiProgressBar').value = range;
    document.getElementById('bmiProgressBar').color = color;
}