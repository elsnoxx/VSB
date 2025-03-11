const infiniteScroll = document.querySelector('ion-infinite-scroll');
const list = document.querySelector('#library-list');  // Zde hledáme správný element pro seznam

infiniteScroll.addEventListener('ionInfinite', (event) => {
    setTimeout(() => {
        getItemsFromLocalStorage();
        event.target.complete();
    }, 500);
});

document.addEventListener("DOMContentLoaded", function () {
    const tabs = document.querySelector("ion-tabs");

    tabs.addEventListener("ionTabsDidChange", (event) => {
        const selectedTab = event.detail.tab;
        if (selectedTab === "library") {
            getItemsFromLocalStorage();
        }
    });
});

function getItemsFromLocalStorage() {
    // Načteme data z localStorage
    const data = localStorage.getItem("bmiHistory");

    // Pokud jsou data k dispozici
    if (data) {
        const parsedData = JSON.parse(data);

        // Zkontroluj, zda data obsahují pole
        if (Array.isArray(parsedData)) {
            list.innerHTML = '';  // Vyčistíme předchozí obsah seznamu

            // Iterace přes každou položku v poli
            for (let i = 0; i < parsedData.length; i++) {
                const item = parsedData[i];

                // Vytvoření nového ion-item pro každou položku
                const ionItem = document.createElement('ion-item');
                const dateObject = new Date(item.date);
                ionItem.innerHTML = `
                            <ion-label>
                                <ion-card style="border: 2px solid #4CAF50; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
                                    <ion-card-header style="background-color: #4CAF50; color: white; border-top-left-radius: 10px; border-top-right-radius: 10px;">
                                        <ion-card-title style="font-size: 1.5em; font-weight: bold; text-align: center;">BMI: ${item.bmi}</ion-card-title>
                                    </ion-card-header>
                                    <ion-card-content style="padding: 20px; font-size: 1.1em;">
                                        <table style="width: 100%; border-collapse: collapse; margin: 10px 0;">
                                            <tr>
                                                <td style="padding: 12px; border: 1px solid #ddd; vertical-align: top; text-align: left;">
                                                    <p><strong>Date:</strong> ${dateObject.toDateString()}</p>
                                                </td>
                                                <td style="padding: 12px; border: 1px solid #ddd; vertical-align: top; text-align: left;">
                                                    <p><strong>Age:</strong> ${item.age}</p>
                                                </td>
                                            </tr>
                                            <tr>
                                                <td style="padding: 12px; border: 1px solid #ddd; vertical-align: top; text-align: left;">
                                                    <p><strong>Height:</strong> ${item.height} cm</p>
                                                </td>
                                                <td style="padding: 12px; border: 1px solid #ddd; vertical-align: top; text-align: left;">
                                                    <p><strong>Weight:</strong> ${item.weight} kg</p>
                                                </td>
                                            </tr>
                                            <tr>
                                                <td style="padding: 12px; border: 1px solid #ddd; vertical-align: top; text-align: left;">
                                                    <p><strong>Gender:</strong> ${item.gender}</p>
                                                </td>
                                                <td style="padding: 12px; border: 1px solid #ddd; vertical-align: top; text-align: left;">
                                                    <p><strong>Metric:</strong> ${item.metric}</p>
                                                </td>
                                            </tr>
                                        </table>
                                    </ion-card-content>
                                </ion-card>
                            </ion-label>
                        `;

                // Přidání ion-item do seznamu
                list.appendChild(ionItem);
            }
        } else {
            console.log("Data nejsou pole.");
        }
    } else {
        const ionItem = document.getElementById('library-list');
        ionItem.innerHTML = "<p>Nebylo nalezeno žádné data v localStorage.</p>";
    }
}

function clearLocalStoradge() {
    localStorage.clear();
    getItemsFromLocalStorage();
}
