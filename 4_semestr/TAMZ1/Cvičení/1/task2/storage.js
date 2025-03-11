window.onload = (event) => {
    addLatestInputs();
};

function saveToSessionStorage(age, height, weight, bmi, gender, metric) {
    const obj = {
        age: age,
        height: height,
        weight: weight,
        bmi: bmi,
        gender: gender,
        metric: metric
    };

    sessionStorage.setItem("lastInput", JSON.stringify(obj));
}

function addLatestInputs() {
    let result = sessionStorage.getItem("lastInput");
    console.log(result);
    if (result !== null) {
        result = JSON.parse(result);
        console.log(result, result.age);
        document.getElementById('age').value = result.age;
        document.getElementById('heightUnitLabel').value = result.height;
        document.getElementById('weightUnitLabel').value = result.weight;
        document.getElementById('gender').value = result.gender;
        document.getElementById('unitSystem').value = result.metric;
        displayBMI(result.bmi);
    }
}

function storeDataToLocalStoradge(height, weight, metric, bmi, gender, age) {
    if (typeof (Storage) !== "undefined" && localStorage != null) {
        const datetime = new Date();
        const obj = {
            date: datetime.getTime(),
            age: age,
            height: height,
            weight: weight,
            bmi: bmi,
            gender: gender,
            metric: metric
        };
        let history = JSON.parse(localStorage.getItem("bmiHistory")) || [];

        history.push(obj);

        localStorage.setItem("bmiHistory", JSON.stringify(history));
    } else {
        alert("before saving you need to enable storadge in you browser")
    }
}

function clearLocalStoradge() {
    localStorage.clear();
    getItemsFromLocalStorage();
}

function clearSessionStoradge() {
    sessionStorage.clear();
}