const infiniteScroll = document.querySelector('ion-infinite-scroll');
const list = document.querySelector('#library-list');
let loadedItems = 0;
const itemsPerLoad = 5;

document.addEventListener("DOMContentLoaded", function () {
    const tabs = document.querySelector("ion-tabs");

    tabs.addEventListener("ionTabsDidChange", (event) => {
        const selectedTab = event.detail.tab;
        if (selectedTab === "library") {
            getItemsFromLocalStorage();
        }
    });
});

// Event listener pro infinite scroll
infiniteScroll.addEventListener('ionInfinite', (event) => {
    setTimeout(() => {
        getItemsFromLocalStorage();
        event.target.complete();
    }, 500);
});

function getItemsFromLocalStorage() {
    const data = localStorage.getItem("bmiHistory");

    if (!data) {
        console.log("Žádná data v localStorage.");
        list.innerHTML = "<p>Nebylo nalezeno žádné data v localStorage.</p>";
        return;
    }

    const parsedData = JSON.parse(data);

    if (!Array.isArray(parsedData) || parsedData.length === 0) {
        console.log("Data nejsou validní pole.");
        list.innerHTML = "<p>Seznam je prázdný.</p>";
        return;
    }

    // Načtení nové sady položek
    const newItems = parsedData.slice(loadedItems, loadedItems + itemsPerLoad);
    loadedItems += newItems.length;

    newItems.forEach(item => {
        const ionItem = document.createElement('ion-item');
        ionItem.innerHTML = generateRow(item);
        list.appendChild(ionItem);
    });

    // // Pokud jsme načetli všechny položky, vypnout infinite scroll
    // if (loadedItems >= parsedData.length) {
    //     infiniteScroll.disabled = true;
    // }
}


function getMetric(metric, type) {
    if (metric === "metric" && type === 'w') {
        return "kg"
    } else if (metric === "metric" && type === 'h') {
        return "cm"
    } else if (metric !== "metric" && type === 'h') {
        return "in"
    } else if (metric !== "metric" && type === 'w') {
        return "lbs"
    }
}

function getProgressBar(bmi) {
    if (bmi < 18.5) {
        return { progress: 0.2, status: "warning" };
    } else if (bmi >= 18.5 && bmi < 24.9) {
        return { progress: 0.5, status: "success" };
    } else if (bmi >= 25 && bmi < 29.9) {
        return { progress: 0.7, status: "warning" };
    } else {
        return { progress: 1, status: "danger" };
    }
}

function generateRow(item){
    const dateObject = new Date(item.date);
    const { progress, status } = getProgressBar(item.bmi);
    let row = `
<ion-label>
    <ion-card class="bmi-card">
        <ion-card-header class="bmi-header">
            <ion-card-title>BMI: ${item.bmi} ${clasificationBMI(item.bmi)}</ion-card-title>
        </ion-card-header>
        <ion-card-content class="bmi-content">
            <table class="bmi-table">
                <tr>
                    <td><strong>Date:</strong> ${dateObject.toDateString()}</td>
                    <td><strong>Age:</strong> ${item.age}</td>
                </tr>
                <tr>
                    <td><strong>Height:</strong> ${item.height} ${getMetric(item.metric, 'h')}</td>
                    <td><strong>Weight:</strong> ${item.weight} ${getMetric(item.metric, 'w')}</td>
                </tr>
                <tr>
                    <td><strong>Gender:</strong> ${item.gender}</td>
                    <td><strong>BMI:</strong> ${item.bmi}</td>
                </tr>
            </table>
            <div id="classification"></div>
            <!-- Progress Bar -->
            <div class="progress-container">
                <ion-progress-bar id="bmiProgressBar" value="${progress}" color="${status}"></ion-progress-bar>
            </div>
        </ion-card-content>
    </ion-card>
</ion-label>`;
    return row;
}