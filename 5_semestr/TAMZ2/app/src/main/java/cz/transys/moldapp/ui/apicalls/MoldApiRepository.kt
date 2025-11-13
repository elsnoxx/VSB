package cz.transys.moldapp.ui.apicalls


class MoldApiRepository {

    suspend fun getAllCars(): List<CarCodeList> {
        return ApiClient.get("MoldPDA/carcode")
    }

    suspend fun getAllCarriers(): List<CarriersList> {
        return ApiClient.get("MoldPDA/molds")
    }
}


// Příklad datového modelu
data class CarCodeList(
    val caR_CODE: String,
    val caR_NAME: String
)

data class CarriersList(
    val codE_VALUE1: String,
    val codE_VALUE2: String
)